#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <laser_geometry/laser_geometry.h>
#include <pointmatcher_ros/point_cloud.h>
#include <pointmatcher/PointMatcher.h>
#include <pointmatcher_ros/transform.h>
#include <tf/transform_listener.h>
#include <nav_msgs/Odometry.h>
#include <fstream>


//#define slam //use own pose estimation insted of pose from tf
//#define icp_test  //debug icp

using namespace PointMatcherSupport;
using namespace std;

typedef PointMatcher<float> PM;
typedef PM::DataPoints DP;
typedef PM::TransformationParameters TP;

//Publishers
ros::Publisher scanCloudPublisher;
ros::Publisher inliersCloudPublisher;
ros::Publisher mapCloudPublisher;
ros::Publisher partialMapCloudPublisher;

#ifdef icp_test
ros::Publisher icpScanPublisher;
#endif
#ifdef slam
double rx=0, ry=0, rth=0; //robot pose
ros::Time lastTime;

void odom_callback(const nav_msgs::Odometry& msg);
void adjust_pose(TP& scanToMapTransform);
#else
//Listener for pose to map transformation
tf::TransformListener* tfListener;
#endif

// ===== Params =====

double cutOffRange = 3.9;
double rateHZ = 5;
string mapFrame = "map";
string poseFrame = "base_scan";
double initPointProbability = 0.8; //0.99
double delPointThreshold = 0;
double inlierPointProbability = 0.99; //0.8
double outlierPointProbability = 0.1; //0.1
bool saveToVTK = false;
int knnRead = 10; //kd-tree //10
int knnRef = 1; //1
double maxScanDensity = 100000; //points per m3
int minPointsToMatch = max(knnRef, knnRead);

double icpDxDyDeviation = 0.2;
double icpYawDeviation = 0.2;

DP mapPoints; //map

// ===== Functions =====

void scan_callback(const sensor_msgs::LaserScan& msg);
pair<DP,DP> filter_transform_map_scan(DP& readPoints); //return: transformed scan and partial map
pair<DP,TP> match_clouds(DP& read, DP& ref); //return: scan outliers and scan to map transform
void update_map(DP& scanOutliers, DP& partialMap);

double calc_point_weight(double probability, double prev=0) {
    return prev + log(probability / (1.0 - probability));
}

// ===== Create filters and transformations =====

PM::DataPointsFilter* normalsFlter(PM::get().DataPointsFilterRegistrar.create(
    "SurfaceNormalDataPointsFilter",
    map_list_of("keepNormals", toParam(0))
               ("keepDensities", toParam(1))
));

PM::Transformation* rigidTrans = PM::get().REG(Transformation).create("RigidTransformation");

PM::DataPointsFilter* densityFlter(PM::get().DataPointsFilterRegistrar.create(
    "MaxDensityDataPointsFilter",
    map_list_of("maxDensity", toParam(maxScanDensity))
));

PM::DataPointsFilter* maxDistFlter(PM::get().DataPointsFilterRegistrar.create(
    "MaxDistDataPointsFilter",
    map_list_of("maxDist", toParam(cutOffRange))
));

PM::DataPointsFilter* minDistFlter(PM::get().DataPointsFilterRegistrar.create(
    "MinDistDataPointsFilter",
    map_list_of("minDist", toParam(cutOffRange))
));

PM::Matcher *matcherRead(PM::get().MatcherRegistrar.create(
    "KDTreeMatcher",
    map_list_of("knn", toParam(knnRead))));

PM::Matcher *matcherReadToTarget(PM::get().MatcherRegistrar.create(
    "KDTreeVarDistMatcher",
    map_list_of("knn", toParam(knnRef))
               ("maxDistField","maxSearchDist") // descriptor name
    ));

#ifdef icp_test
PM::ICP icp;
#endif

// ===== Main =====

int main(int argc, char** argv)
{
    ros::init(argc, argv, "features_maintainer_node");

#ifdef icp_test
    // Load ICP conf from file
    ifstream conf;
    conf.open("conf.yaml");
    icp.loadFromYaml(conf);
#endif

    //Create node and topics
    ros::NodeHandle node("~");

    scanCloudPublisher = node.advertise<sensor_msgs::PointCloud2>("/scan_cloud", 10);
    inliersCloudPublisher = node.advertise<sensor_msgs::PointCloud2>("/matched_cloud", 10);
    mapCloudPublisher = node.advertise<sensor_msgs::PointCloud2>("/map_cloud", 10);
    partialMapCloudPublisher = node.advertise<sensor_msgs::PointCloud2>("/partial_map_cloud", 10);
    ros::Subscriber scanSubscriber = node.subscribe("/scan", 1, scan_callback);

#ifdef icp_test
    icpScanPublisher = node.advertise<sensor_msgs::PointCloud2>("/icp_scan_cloud", 10);
#endif
#ifdef slam
    lastTime = ros::Time::now();
    ros::Subscriber odomSubscriber = node.subscribe("/odom", 1, odom_callback);
#else
    tfListener = new tf::TransformListener();
#endif

    ros::Rate rate(rateHZ);
    while (ros::ok()) {        
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
};

#ifdef slam
void odom_callback(const nav_msgs::Odometry& msg)
{
    ros::Time currentTime = msg.header.stamp;
    double vx = msg.twist.twist.linear.x;
    double vy = msg.twist.twist.linear.y;
    double vth = msg.twist.twist.angular.z;

    double dt = (currentTime - lastTime).toSec();
    double delta_x = (vx * cos(rth) - vy * sin(rth)) * dt;
    double delta_y = (vx * sin(rth) + vy * cos(rth)) * dt;
    double delta_th = vth * dt;

    rx += delta_x;
    ry += delta_y;
    rth += delta_th;

    lastTime = currentTime;
}

void adjust_pose(TP& scanToMapTransform) {

    //Calc corrections from matrix
    double dx = scanToMapTransform(0, 3);
    double dy = scanToMapTransform(1, 3);

    double pitch = -asin(scanToMapTransform(0,2));
    double roll = atan2(scanToMapTransform(2,1), scanToMapTransform(2,2));
    double yaw = atan2(scanToMapTransform(1,0) / cos(pitch), scanToMapTransform(0,0) / cos(pitch));

    //Apply corrections if they in bounds
    if (abs(dx) > icpDxDyDeviation || abs(dy) > icpDxDyDeviation || abs(yaw) > icpYawDeviation)
        return;

    rx+=dx;
    ry+=dy;
    rth+=yaw;
}

#endif

void scan_callback(const sensor_msgs::LaserScan& msg)
{

     //========== Convert scan points cloud ==========

    //Convert scan to ROS points cloud
    sensor_msgs::PointCloud2 cloud;
    laser_geometry::LaserProjection projector;
    projector.projectLaser(msg, cloud, cutOffRange);

    //Convert ROS points cloud to libpoTP scanToMapTransformintmatcher cloud
    DP readPoints = PointMatcher_ros::rosMsgToPointMatcherCloud<float>(cloud);
    //if no poits in scan - return
    if (readPoints.getNbPoints()<minPointsToMatch) return;

    //========== Transform scan ==========

    //filter density of scan, transform to pose
    //select map points with some radius, remove them from mapPoints
    pair<DP,DP> clouds = filter_transform_map_scan(readPoints);
    DP readTransformedPoints = clouds.first, partialMap = clouds.second;

    //========== Create map ==========

    //Create map if empty
    if (mapPoints.getNbPoints() == 0) {
        mapPoints = DP(readTransformedPoints);
        PM::Matrix weightsMap = PM::Matrix::Constant(1, mapPoints.features.cols(), calc_point_weight(initPointProbability));
        mapPoints.addDescriptor("weights", weightsMap);
    }

    //========== Update map ==========

    //If not enough points in partial map to match add all points from scan to full map
    else if (partialMap.getNbPoints() < minPointsToMatch){
        update_map(readTransformedPoints, partialMap);
    }
    else {
        //Find matches of transformed scan and selected points of map
        pair<DP,TP> outliersAndParams = match_clouds(readTransformedPoints, partialMap);
        DP readOutliers = outliersAndParams.first;
        TP scanToMapTransform = outliersAndParams.second;

        //Update weights of map points, append scan outliers
        update_map(readOutliers, partialMap);

#ifdef slam
#ifdef icp_test
        //Correct pose with transform params from scan to map
        adjust_pose(scanToMapTransform);
#endif
#endif
    }
}





pair<DP,DP> filter_transform_map_scan(DP& readPoints) {

    //========== Get and check pose transformation params ==========

    TP robotPoseTrans;

#ifdef slam
    nav_msgs::Odometry rosOdom;
    rosOdom.pose.pose.position.x = rx;
    rosOdom.pose.pose.position.y = ry;

    tf::Quaternion quatTF = tf::createQuaternionFromYaw(rth);
    geometry_msgs::Quaternion quatROS;
    tf::quaternionTFToMsg(quatTF, quatROS);
    rosOdom.pose.pose.orientation = quatROS;

    robotPoseTrans = PointMatcher_ros::odomMsgToEigenMatrix<float>(rosOdom);
#else
    try {
        //Get transformation of new points cloud to map
        robotPoseTrans = PointMatcher_ros::transformListenerToEigenMatrix<float>(*tfListener, mapFrame, poseFrame, ros::Time(0));
    }
    catch (tf2::TransformException& ex)
    {
        ROS_WARN("%s", ex.what());
        return make_pair(DP(), DP());
    }

    if (!rigidTrans->checkParameters(robotPoseTrans)) {
        std::cout << "WARNING: T does not represent a valid rigid transformation"<< std::endl;
        return make_pair(DP(), DP());
    }
#endif

    //========== Filter and transform scan ==========

    //Calc densities descriptor
    readPoints = normalsFlter->filter(readPoints);

    //Filter scan by density
    readPoints = densityFlter->filter(readPoints);

    readPoints.removeDescriptor("densities");

    // Compute the transformation from pose to map
    DP readTransformedPoints =  rigidTrans->compute(readPoints, robotPoseTrans);
    //Publish to ROS raw scan
    scanCloudPublisher.publish(
        PointMatcher_ros::pointMatcherCloudToRosMsg<float>(readTransformedPoints, mapFrame, ros::Time(0)));


    //========== Filter and transform map ==========

    //Transform map to origin
    TP robotPoseTransInv = robotPoseTrans.inverse();
    DP mapAtOrigin = rigidTrans->compute(mapPoints, robotPoseTransInv);

    //Filter map by radius - get only closest points
    DP partialMap = maxDistFlter->filter(mapAtOrigin);

    if (partialMap.getNbPoints() == 0)
        return make_pair(readTransformedPoints, DP());

    partialMap = rigidTrans->compute(partialMap, robotPoseTrans);

    //Filter map by radius - get least of map
    mapPoints = minDistFlter->filter(mapAtOrigin);
    mapPoints = rigidTrans->compute(mapPoints, robotPoseTrans);

    //Publish to ROS partial map
    partialMapCloudPublisher.publish(
        PointMatcher_ros::pointMatcherCloudToRosMsg<float>(partialMap, mapFrame, ros::Time(0)));

    return make_pair(readTransformedPoints, partialMap);
}





pair<DP,TP> match_clouds(DP& read, DP& ref) {

    int readPtsCount = read.getNbPoints();
    int refPtsCount = ref.getNbPoints();

    TP scanToMapTransform;

#ifdef icp_test
    // ========== Match points with ICP ==========
    // Don't work correct, only for debug. Below using not transformed cloud

    // Compute the transformation to express readPoints in mapPoints
    scanToMapTransform = icp(read, ref);

    // Transform readPoints to express it in mapPoints
    DP readPointsTransformed(read);
    icp.transformations.apply(readPointsTransformed, scanToMapTransform);

    //Publish to ROS icp transformed scan
    icpScanPublisher.publish(
        PointMatcher_ros::pointMatcherCloudToRosMsg<float>(readPointsTransformed, mapFrame, ros::Time(0)));
#endif

    // ========== Find matches ==========

    // Matcher to find closest points on read scan
    matcherRead->init(read);

    // Find clothest points on read
    PM::Matches readMatches(knnRead, readPtsCount);
    readMatches = matcherRead->findClosests(read);

    // For every point set maximum search distanse as square root of distanse to
    // farest point, found by matcherRead
    PM::Matrix maxSearchDist = readMatches.dists.colwise().maxCoeff().cwiseSqrt();
    read.addDescriptor("maxSearchDist", maxSearchDist);

    // Matcher to match points on both scans with limit to max distance
    matcherReadToTarget->init(ref);

    // Find matches from read to ref
    PM::Matches refMatches(knnRef, refPtsCount);
        refMatches = matcherReadToTarget->findClosests(read);

    // Add new descriptors to select inliers and outliers
    PM::Matrix inliersRead = PM::Matrix::Zero(1, read.getNbPoints());
    read.addDescriptor("inliers", inliersRead);
    PM::Matrix inliersRef = PM::Matrix::Zero(1, ref.getNbPoints());
    ref.addDescriptor("inliers", inliersRef);

    // Get view to edit inliers
    DP::View inlierRead = read.getDescriptorViewByName("inliers");
    DP::View inlierRef = ref.getDescriptorViewByName("inliers");

    // For every point
    for (int i = 0; i < readPtsCount; i++) {
      // For every match
      for (int k = 0; k < knnRef; k++) {

        if (refMatches.dists(k, i) != PM::Matches::InvalidDist) {
          // Set inlier descriptor to both points
          inlierRead(0, i) = 1.0;
          inlierRef(0, refMatches.ids(k, i)) = 1.0;
        }

      }
    }

    // ========== Filter outliers ==========

    // Copy read outliers and inliers to new clouds
    DP readOutliers = read.createSimilarEmpty();
    int readOutliersCount = 0;
    int readInliersCount = 0;
    for (int i = 0; i < readPtsCount; i++) {
        if (inlierRead(0,i)==0) {
            readOutliers.features.col(readOutliersCount) = read.features.col(i);
            readOutliers.descriptors.col(readOutliersCount) = read.descriptors.col(i);
            readOutliersCount++;
        }
        else {
            readInliersCount++;
        }
    }
    readOutliers.conservativeResize(readOutliersCount);

    // Copy map outliers and inliers to new clouds
    DP refInliers = ref.createSimilarEmpty();
    int refOutliersCount = 0;
    int refInliersCount = 0;
    for (int i = 0; i < refPtsCount; i++) {
        if (inlierRef(0,i)==0) {
            refOutliersCount++;
        }
        else {
            refInliers.features.col(refInliersCount) = ref.features.col(i);
            refInliers.descriptors.col(refInliersCount) = ref.descriptors.col(i);
            refInliersCount++;
        }
    }
    refInliers.conservativeResize(refInliersCount);

    // ========== Update map ==========

    sensor_msgs::PointCloud2 inliersCloud = PointMatcher_ros::pointMatcherCloudToRosMsg<float>(refInliers, mapFrame, ros::Time(0));
    inliersCloudPublisher.publish(inliersCloud);

    // ========== Debug ==========

    stringstream ss;
    ss << endl <<
          "read: all=" << read.getNbPoints() <<
          ", in=" << readInliersCount <<
          ", out=" << readOutliersCount << endl <<
          "ref:  all=" << ref.getNbPoints() <<
          ", in=" << refInliersCount <<
          ", out=" << refOutliersCount << endl <<
          "map_least:  all=" << mapPoints.getNbPoints();
    ROS_INFO("%s", ss.str().c_str());

    if (saveToVTK) {
        read.save("read_orig.vtk");
        mapPoints.save("map.vtk");
        ref.save("ref.vtk");
        refInliers.save("ref_inliers.vtk");
        readOutliers.save("read_outliers.vtk");
    }

    return make_pair(readOutliers, scanToMapTransform);

}





void update_map(DP& scanOutliers, DP& partialMap) {

    int mapSize = partialMap.getNbPoints();

    // ========== Update weights of map  ==========

    //if points from map were not matched (because them not enough), do not update their weights
    if (mapSize>=minPointsToMatch) {

        DP::View mapWeightsDesc = partialMap.getDescriptorViewByName("weights");
        DP::View mapInliersDesc = partialMap.getDescriptorViewByName("inliers");

        // For map inliers add weight, for outliers sub weight
        for (int i=0; i<mapSize; i++) {
            if (mapInliersDesc(0,i) == 1.0)
                mapWeightsDesc(0,i) += calc_point_weight(inlierPointProbability);
            else
                mapWeightsDesc(0,i) += calc_point_weight(outlierPointProbability);
        }

        //Remove points from map that weight less than threshold
        DP newMapPoints = partialMap.createSimilarEmpty();
        int count = 0;
        for (int i = 0; i < mapSize; i++) {
            if (mapWeightsDesc(0,i)>delPointThreshold) {
                newMapPoints.features.col(count) = partialMap.features.col(i);
                newMapPoints.descriptors.col(count) = partialMap.descriptors.col(i);
                count++;
            }
        }
        newMapPoints.conservativeResize(count);

        //Add partial map to least of map
        newMapPoints.removeDescriptor("inliers");
        mapPoints.concatenate(newMapPoints);
    }

    // ========== Add scan outliers to map  ==========

    PM::Matrix weightsScanOutliers = PM::Matrix::Constant(1, scanOutliers.getNbPoints(), calc_point_weight(initPointProbability));
    scanOutliers.addDescriptor("weights", weightsScanOutliers);
    scanOutliers.removeDescriptor("inliers");
    mapPoints.concatenate(scanOutliers);

    // ========== Publish to ROS ==========

    sensor_msgs::PointCloud2 mapCloud = PointMatcher_ros::pointMatcherCloudToRosMsg<float>(mapPoints, mapFrame, ros::Time(0));
    mapCloudPublisher.publish(mapCloud);


}
