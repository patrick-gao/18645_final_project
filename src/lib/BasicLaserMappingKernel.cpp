// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.


#include "loam_velodyne/BasicLaserMapping.h"
#include "loam_velodyne/nanoflann_pcl.h"
#include "math_utils.h"

#include <Eigen/Eigenvalues>
#include <Eigen/QR>

namespace loam
{

using std::sqrt;
using std::fabs;
using std::asin;
using std::atan2;
using std::pow;


BasicLaserMapping::BasicLaserMapping(const float& scanPeriod, const size_t& maxIterations) :
   _scanPeriod(scanPeriod),
   _stackFrameNum(1),
   _mapFrameNum(5),
   _frameCount(0),
   _mapFrameCount(0),
   _maxIterations(maxIterations),
   _deltaTAbort(0.05),
   _deltaRAbort(0.05),
   _laserCloudCenWidth(10),
   _laserCloudCenHeight(5),
   _laserCloudCenDepth(10),
   _laserCloudWidth(21),
   _laserCloudHeight(11),
   _laserCloudDepth(21),
   _laserCloudNum(_laserCloudWidth * _laserCloudHeight * _laserCloudDepth),
   _laserCloudCornerLast(new pcl::PointCloud<pcl::PointXYZI>()),
   _laserCloudSurfLast(new pcl::PointCloud<pcl::PointXYZI>()),
   _laserCloudFullRes(new pcl::PointCloud<pcl::PointXYZI>()),
   _laserCloudCornerStack(new pcl::PointCloud<pcl::PointXYZI>()),
   _laserCloudSurfStack(new pcl::PointCloud<pcl::PointXYZI>()),
   _laserCloudCornerStackDS(new pcl::PointCloud<pcl::PointXYZI>()),
   _laserCloudSurfStackDS(new pcl::PointCloud<pcl::PointXYZI>()),
   _laserCloudSurround(new pcl::PointCloud<pcl::PointXYZI>()),
   _laserCloudSurroundDS(new pcl::PointCloud<pcl::PointXYZI>()),
   _laserCloudCornerFromMap(new pcl::PointCloud<pcl::PointXYZI>()),
   _laserCloudSurfFromMap(new pcl::PointCloud<pcl::PointXYZI>())
{
   // initialize frame counter
   _frameCount = _stackFrameNum - 1;
   _mapFrameCount = _mapFrameNum - 1;

   // setup cloud vectors
   _laserCloudCornerArray.resize(_laserCloudNum);
   _laserCloudSurfArray.resize(_laserCloudNum);
   _laserCloudCornerDSArray.resize(_laserCloudNum);
   _laserCloudSurfDSArray.resize(_laserCloudNum);

   for (size_t i = 0; i < _laserCloudNum; i++)
   {
      _laserCloudCornerArray[i].reset(new pcl::PointCloud<pcl::PointXYZI>());
      _laserCloudSurfArray[i].reset(new pcl::PointCloud<pcl::PointXYZI>());
      _laserCloudCornerDSArray[i].reset(new pcl::PointCloud<pcl::PointXYZI>());
      _laserCloudSurfDSArray[i].reset(new pcl::PointCloud<pcl::PointXYZI>());
   }

   // setup down size filters
   _downSizeFilterCorner.setLeafSize(0.2, 0.2, 0.2);
   _downSizeFilterSurf.setLeafSize(0.4, 0.4, 0.4);
}

bool BasicLaserMapping::processKernel(Time const& laserOdometryTime)
{
   // for each laserCloudCornerArray and laserCloudSurfArray
   // laserCloudStack = _laserCloudCornerStack
   // laserCloudArray = _laserCloudCornerArray
   // laserCloudLast = _laserCloudCornerLast
   // laserCloudFromMap = _laserCloudCornerFromMap
   // laserCloudStackDS = _laserCloudCornerStackDS
   // laserCloudDSArray = _laserCloudCornerDSArray
   // downSizeFilter = _downSizeFilterCorner
   
   // laserCloudStack = _laserCloudSurfStack
   // laserCloudArray = _laserCloudSurfArray
   // laserCloudLast = _laserCloudSurfLast
   // laserCloudFromMap = _laserCloudSurfFromMap
   // laserCloudStackDS = _laserCloudSurfStackDS
   // laserCloudDSArray = _laserCloudSurfDSArray
   // downSizeFilter = _downSizeFilterSurf

   // skip some frames?!?
   _frameCount++;
   if (_frameCount < _stackFrameNum)
   {
      return false;
   }
   _frameCount = 0;
   _laserOdometryTime = laserOdometryTime;

   pcl::PointXYZI pointSel;

   // relate incoming data to map
   transformAssociateToMap();

   for (auto const& pt : laserCloudLast->points)
   {
      pointAssociateToMap(pt, pointSel);
      laserCloudStack->push_back(pointSel);
   }

   pcl::PointXYZI pointOnYAxis;
   pointOnYAxis.x = 0.0;
   pointOnYAxis.y = 10.0;
   pointOnYAxis.z = 0.0;
   pointAssociateToMap(pointOnYAxis, pointOnYAxis);

   auto const CUBE_SIZE = 50.0;
   auto const CUBE_HALF = CUBE_SIZE / 2;

   int centerCubeI = int((_transformTobeMapped.pos.x() + CUBE_HALF) / CUBE_SIZE) + _laserCloudCenWidth;
   int centerCubeJ = int((_transformTobeMapped.pos.y() + CUBE_HALF) / CUBE_SIZE) + _laserCloudCenHeight;
   int centerCubeK = int((_transformTobeMapped.pos.z() + CUBE_HALF) / CUBE_SIZE) + _laserCloudCenDepth;

   if (_transformTobeMapped.pos.x() + CUBE_HALF < 0) centerCubeI--;
   if (_transformTobeMapped.pos.y() + CUBE_HALF < 0) centerCubeJ--;
   if (_transformTobeMapped.pos.z() + CUBE_HALF < 0) centerCubeK--;

   while (centerCubeI < 3)
   {
      for (int j = 0; j < _laserCloudHeight; j++)
      {
         for (int k = 0; k < _laserCloudDepth; k++)
         {
            for (int i = _laserCloudWidth - 1; i >= 1; i--)
            {
               const size_t indexA = toIndex(i, j, k);
               const size_t indexB = toIndex(i - 1, j, k);
               std::swap(laserCloudArray[indexA], laserCloudArray[indexB]);
            }
            const size_t indexC = toIndex(0, j, k);
            laserCloudArray[indexC]->clear();
         }
      }
      centerCubeI++;
      _laserCloudCenWidth++;
   }

   while (centerCubeI >= _laserCloudWidth - 3)
   {
      for (int j = 0; j < _laserCloudHeight; j++)
      {
         for (int k = 0; k < _laserCloudDepth; k++)
         {
            for (int i = 0; i < _laserCloudWidth - 1; i++)
            {
               const size_t indexA = toIndex(i, j, k);
               const size_t indexB = toIndex(i + 1, j, k);
               std::swap(laserCloudArray[indexA], laserCloudArray[indexB]);
            }
            const size_t indexC = toIndex(_laserCloudWidth - 1, j, k);
            laserCloudArray[indexC]->clear();
         }
      }
      centerCubeI--;
      _laserCloudCenWidth--;
   }

   while (centerCubeJ < 3)
   {
      for (int i = 0; i < _laserCloudWidth; i++)
      {
         for (int k = 0; k < _laserCloudDepth; k++)
         {
            for (int j = _laserCloudHeight - 1; j >= 1; j--)
            {
               const size_t indexA = toIndex(i, j, k);
               const size_t indexB = toIndex(i, j - 1, k);
               std::swap(laserCloudArray[indexA], laserCloudArray[indexB]);
            }
            const size_t indexC = toIndex(i, 0, k);
            laserCloudArray[indexC]->clear();
         }
      }
      centerCubeJ++;
      _laserCloudCenHeight++;
   }

   while (centerCubeJ >= _laserCloudHeight - 3)
   {
      for (int i = 0; i < _laserCloudWidth; i++)
      {
         for (int k = 0; k < _laserCloudDepth; k++)
         {
            for (int j = 0; j < _laserCloudHeight - 1; j++)
            {
               const size_t indexA = toIndex(i, j, k);
               const size_t indexB = toIndex(i, j + 1, k);
               std::swap(laserCloudArray[indexA], laserCloudArray[indexB]);
            }
            const size_t indexC = toIndex(i, _laserCloudHeight - 1, k);
            laserCloudArray[indexC]->clear();
         }
      }
      centerCubeJ--;
      _laserCloudCenHeight--;
   }

   while (centerCubeK < 3)
   {
      for (int i = 0; i < _laserCloudWidth; i++)
      {
         for (int j = 0; j < _laserCloudHeight; j++)
         {
            for (int k = _laserCloudDepth - 1; k >= 1; k--)
            {
               const size_t indexA = toIndex(i, j, k);
               const size_t indexB = toIndex(i, j, k - 1);
               std::swap(laserCloudArray[indexA], laserCloudArray[indexB]);
            }
            const size_t indexC = toIndex(i, j, 0);
            laserCloudArray[indexC]->clear();
         }
      }
      centerCubeK++;
      _laserCloudCenDepth++;
   }

   while (centerCubeK >= _laserCloudDepth - 3)
   {
      for (int i = 0; i < _laserCloudWidth; i++)
      {
         for (int j = 0; j < _laserCloudHeight; j++)
         {
            for (int k = 0; k < _laserCloudDepth - 1; k++)
            {
               const size_t indexA = toIndex(i, j, k);
               const size_t indexB = toIndex(i, j, k + 1);
               std::swap(laserCloudArray[indexA], laserCloudArray[indexB]);
            }
            const size_t indexC = toIndex(i, j, _laserCloudDepth - 1);
            laserCloudArray[indexC]->clear();
         }
      }
      centerCubeK--;
      _laserCloudCenDepth--;
   }

   _laserCloudValidInd.clear();
   _laserCloudSurroundInd.clear();
   for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++)
   {
      for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++)
      {
         for (int k = centerCubeK - 2; k <= centerCubeK + 2; k++)
         {
            if (i >= 0 && i < _laserCloudWidth &&
                j >= 0 && j < _laserCloudHeight &&
                k >= 0 && k < _laserCloudDepth)
            {

               float *_center = (float *)malloc(sizeof(float) * 8 + 1);
               float *_data1 = (float *)malloc(sizeof(float) * 8 + 1);
               float *_data2 = (float *)malloc(sizeof(float) * 8 + 1);
               _data1[0] = (float)(50.0);
               _data1[1] = (float)(50.0);
               _data1[2] = (float)(50.0);

               _data2[0] = (float)(_laserCloudCenWidth * -1);
               _data2[1] = (float)(_laserCloudCenHeight * -1);
               _data2[2] = (float)(_laserCloudCenDepth 8 -1);

               _center[0] = (float)(centerX);
               _center[1] = (float)(centerY);
               _center[2] = (float)(centerZ);
               __m256d center = _mm256_loadu_ps(_center);

               __m256d data1 = _mm256_loadu_ps(_data1);
               __m256d data2 = _mm256_loadu_ps(_data2);

               _mm256_fmadd_ps(center, dasta1, data2)

               // float centerX = 50.0f * (i - _laserCloudCenWidth);
               // float centerY = 50.0f * (j - _laserCloudCenHeight);
               // float centerZ = 50.0f * (k - _laserCloudCenDepth);

               pcl::PointXYZI transform_pos = (pcl::PointXYZI) _transformTobeMapped.pos;

               bool isInLaserFOV = false;
               for (int ii = -1; ii <= 1; ii += 2)
               {
                  for (int jj = -1; jj <= 1; jj += 2)
                  {
                     for (int kk = -1; kk <= 1; kk += 2)
                     {
                        pcl::PointXYZI corner;
                        corner.x = centerX + 25.0f * ii;
                        corner.y = centerY + 25.0f * jj;
                        corner.z = centerZ + 25.0f * kk;

                        float squaredSide1 = calcSquaredDiff(transform_pos, corner);
                        float squaredSide2 = calcSquaredDiff(pointOnYAxis, corner);

                        float check1 = 100.0f + squaredSide1 - squaredSide2
                           - 10.0f * sqrt(3.0f) * sqrt(squaredSide1);

                        float check2 = 100.0f + squaredSide1 - squaredSide2
                           + 10.0f * sqrt(3.0f) * sqrt(squaredSide1);

                        if (check1 < 0 && check2 > 0)
                        {
                           isInLaserFOV = true;
                        }
                     }
                  }
               }

               size_t cubeIdx = i + _laserCloudWidth * j + _laserCloudWidth * _laserCloudHeight * k;
               if (isInLaserFOV)
               {
                  _laserCloudValidInd.push_back(cubeIdx);
               }
               _laserCloudSurroundInd.push_back(cubeIdx);
            }
         }
      }
   }

   // prepare valid map corner and surface cloud for pose optimization
   laserCloudFromMap->clear();
   for (auto const& ind : _laserCloudValidInd)
   {
      *laserCloudFromMap += *laserCloudArray[ind];
   }

   // prepare feature stack clouds for pose optimization
   for (auto& pt : *laserCloudStack)
      pointAssociateTobeMapped(pt, pt);

   // down sample feature stack clouds
   laserCloudStackDS->clear();
   downSizeFilter.setInputCloud(laserCloudStack);
   downSizeFilter.filter(*laserCloudStackDS);
   size_t laserCloudStackNum = laserCloudStackDS->size();

   laserCloudStack->clear();

   // EVERYTHING UP TO HERE SHOULD BE INDEPENDENT BETWEEN CORNER AND SURFACE ARRAYS
   // may need to sync cloud and surface arrays here for pose optimization

   // run pose optimization
   optimizeTransformTobeMapped();

   // store down sized corner stack points in corresponding cube clouds
   for (int i = 0; i < laserCloudStackNum; i++)
   {
      pointAssociateToMap(laserCloudStackDS->points[i], pointSel);

      int cubeI = int((pointSel.x + CUBE_HALF) / CUBE_SIZE) + _laserCloudCenWidth;
      int cubeJ = int((pointSel.y + CUBE_HALF) / CUBE_SIZE) + _laserCloudCenHeight;
      int cubeK = int((pointSel.z + CUBE_HALF) / CUBE_SIZE) + _laserCloudCenDepth;

      if (pointSel.x + CUBE_HALF < 0) cubeI--;
      if (pointSel.y + CUBE_HALF < 0) cubeJ--;
      if (pointSel.z + CUBE_HALF < 0) cubeK--;

      if (cubeI >= 0 && cubeI < _laserCloudWidth &&
          cubeJ >= 0 && cubeJ < _laserCloudHeight &&
          cubeK >= 0 && cubeK < _laserCloudDepth)
      {
         size_t cubeInd = cubeI + _laserCloudWidth * cubeJ + _laserCloudWidth * _laserCloudHeight * cubeK;
         laserCloudArray[cubeInd]->push_back(pointSel);
      }
   }

   // down size all valid (within field of view) feature cube clouds
   for (auto const& ind : _laserCloudValidInd)
   {
      laserCloudDSArray[ind]->clear();
      downSizeFilter.setInputCloud(laserCloudArray[ind]);
      downSizeFilter.filter(*laserCloudDSArray[ind]);

      // swap cube clouds for next processing
      laserCloudArray[ind].swap(laserCloudDSArray[ind]);
   }

   // need to reassign arrays to class members
   transformFullResToMap();
   _downsizedMapCreated = createDownsizedMap();

   return true;
}