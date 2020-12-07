#include <ros/ros.h>
#include <ros/console.h>
#include "loam_velodyne/LaserMapping.h"


//timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

/** Main node entry point. */
int main(int argc, char **argv)
{
  ros::init(argc, argv, "laserMapping");
  ros::NodeHandle node;
  ros::NodeHandle privateNode("~");

  loam::LaserMapping laserMapping(0.1);

  if (laserMapping.setup(node, privateNode)) {
    // initialization successful
    unsigned long long t0 = rdtsc();
    
    laserMapping.spin();
    
    unsigned long long t1 = rdtsc();
    
    ROS_INFO_STREAM(t1-t0);
  }

  return 0;
}
