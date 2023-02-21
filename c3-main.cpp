
#include <carla/client/Client.h>
#include <carla/client/ActorBlueprint.h>
#include <carla/client/BlueprintLibrary.h>
#include <carla/client/Map.h>
#include <carla/geom/Location.h>
#include <carla/geom/Transform.h>
#include <carla/client/Sensor.h>
#include <carla/sensor/data/LidarMeasurement.h>
#include <thread>

#include <carla/client/Vehicle.h>

//pcl code
//#include "render/render.h"

namespace cc = carla::client;
namespace cg = carla::geom;
namespace csd = carla::sensor::data;

using namespace std::chrono_literals;
using namespace std::string_literals;

using namespace std;




#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include "helper.h"
#include <sstream>
#include <chrono> 
#include <ctime> 
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/console/time.h>  // TicToc
#include <Eigen/Core>
#include <Eigen/SVD>
using namespace Eigen;

struct Pair{

	Point p1;
	Point p2;

	Pair(Point setP1, Point setP2)
		: p1(setP1), p2(setP2){}
};


PointCloudT pclCloud;
cc::Vehicle::Control control;
std::chrono::time_point<std::chrono::system_clock> currentTime;
vector<ControlState> cs;

bool refresh_view = false;
bool save_map = false;
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* viewer)
{

  	//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *>(viewer_void);
	if (event.getKeySym() == "Right" && event.keyDown()){
		cs.push_back(ControlState(0, -0.02, 0));
  	}
	else if (event.getKeySym() == "Left" && event.keyDown()){
		cs.push_back(ControlState(0, 0.02, 0)); 
  	}
  	if (event.getKeySym() == "Up" && event.keyDown()){
		cs.push_back(ControlState(0.1, 0, 0));
  	}
	else if (event.getKeySym() == "Down" && event.keyDown()){
		cs.push_back(ControlState(-0.1, 0, 0)); 
  	}
	if(event.getKeySym() == "a" && event.keyDown()){
		refresh_view = true;
	}
  	//Added command to save the scanned map
  	if(event.getKeySym() == "s" && event.keyDown()){
		save_map = true;
	}
}

void Accuate(ControlState response, cc::Vehicle::Control& state){

	if(response.t > 0){
		if(!state.reverse){
			state.throttle = min(state.throttle+response.t, 1.0f);
		}
		else{
			state.reverse = false;
			state.throttle = min(response.t, 1.0f);
		}
	}
	else if(response.t < 0){
		response.t = -response.t;
		if(state.reverse){
			state.throttle = min(state.throttle+response.t, 1.0f);
		}
		else{
			state.reverse = true;
			state.throttle = min(response.t, 1.0f);

		}
	}
	state.steer = min( max(state.steer+response.s, -1.0f), 1.0f);
	state.brake = response.b;
}

void drawCar(Pose pose, int num, Color color, double alpha, pcl::visualization::PCLVisualizer::Ptr& viewer){

	BoxQ box;
	box.bboxTransform = Eigen::Vector3f(pose.position.x, pose.position.y, 0);
    box.bboxQuaternion = getQuaternion(pose.rotation.yaw);
    box.cube_length = 4;
    box.cube_width = 2;
    box.cube_height = 2;
	renderBox(viewer, box, num, color, alpha);
}

//ICP algorithm from pcl libarary pcl/registration/icp.h
Eigen::Matrix4d ICP(PointCloudT::Ptr target, PointCloudT::Ptr source, Pose startingPose, int iterations){

  	Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity ();
  	Eigen::Matrix4d trans = transform3D(startingPose.rotation.yaw, startingPose.rotation.pitch, startingPose.rotation.roll, startingPose.position.x, startingPose.position.y, startingPose.position.z);
    PointCloudT::Ptr source_tans(new PointCloudT);
  	pcl::transformPointCloud (*source, *source_tans,trans);
  	pcl::IterativeClosestPoint<PointT, PointT> icp;
	icp.setInputSource(source_tans);
	icp.setInputTarget(target);
  	icp.setMaximumIterations (iterations);
  	icp.setMaxCorrespondenceDistance (2.0);
    PointCloudT::Ptr Final(new PointCloudT);
    icp.align(*Final);
  	if(icp.hasConverged()){
    	transformation_matrix = icp.getFinalTransformation().cast<double>();
      	transformation_matrix = transformation_matrix*trans;
      	return transformation_matrix;
    }
  	return transformation_matrix*trans;
}

//NDT algorithm from pcl libarary pcl/registration/ndt.h
Eigen::Matrix4d NDT(pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt, PointCloudT::Ptr source, Pose startingPose, int iterations){

  	Eigen::Matrix4f init_guess = transform3D(startingPose.rotation.yaw, startingPose.rotation.pitch, startingPose.rotation.roll, startingPose.position.x, startingPose.position.y, startingPose.position.z).cast<float>();
	
  	ndt.setMaximumIterations (iterations);
	ndt.setInputSource (source);
  	
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ndt (new pcl::PointCloud<pcl::PointXYZ>);
  	ndt.align (*cloud_ndt, init_guess);

	Eigen::Matrix4d transformation_matrix = ndt.getFinalTransformation ().cast<double>();

	return transformation_matrix;
}



//Function to pair points of target and source using associations vector
//Used in ICP1 function (ICP from scratch algorithm)
vector<Pair> PairPoints(vector<int> associations, PointCloudT::Ptr target, PointCloudT::Ptr source){

	vector<Pair> pairs;
  	int index = 0;
  	for(PointT point:source->points){
      if(associations[index]>=0){
        PointT association =(*target)[associations[index]];
        pairs.push_back(Pair(Point(point.x, point.y,point.z), Point(association.x, association.y,point.z)));
      }
      index++;
    }
	return pairs;
}


//ICP algorithm from scratch
Eigen::Matrix4d ICP1( PointCloudT::Ptr target, PointCloudT::Ptr source, Pose startingPose, int iterations,pcl::KdTreeFLANN<PointT> &kdtree){

  	Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
  	Eigen::Matrix4d final_transformation_matrix = Eigen::Matrix4d::Identity();
  	Eigen::Matrix4d initTransform = transform3D(startingPose.rotation.yaw, startingPose.rotation.pitch, startingPose.rotation.roll, startingPose.position.x, startingPose.position.y, startingPose.position.z);

  	PointCloudT::Ptr source_trans(new PointCloudT);
  	PointCloudT::Ptr source_trans1(new PointCloudT);
  
  	pcl::transformPointCloud(*source,*source_trans,initTransform);
  	pcl::transformPointCloud(*source,*source_trans1,initTransform);
  
  	for(int i =0;i<iterations;i++){
      		pcl::transformPointCloud(*source_trans1,*source_trans,final_transformation_matrix);
      		//Making associations between source and target point clouds
            vector<int> associations;
            for(PointT p:source_trans->points){
                vector<int> pointIdxRadiusSearch;
                vector<float> pointRadiusSquaredDistance;
                if(kdtree.radiusSearch(p,1.0,pointIdxRadiusSearch,pointRadiusSquaredDistance)>0){
                    associations.push_back(pointIdxRadiusSearch[0]);            
                }
                else{
                    associations.push_back(-1);
                }
            }
			//Pairing the associated points
            vector<Pair> pairs = PairPoints(associations,target,source_trans);
            int index = 0;
            Eigen::MatrixXd P = Eigen::MatrixXd::Zero(3,1);
            Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(3,1);
            //Calculating centroid for target(P) and source(Q)
      		for(Pair p:pairs){
                P(0,0) += p.p1.x;
                Q(0,0) += p.p2.x;
                P(1,0) += p.p1.y;
                Q(1,0) += p.p2.y;
                P(2,0) += p.p1.z;
                Q(2,0) += p.p2.z;
                index++;        
              }
            P(0,0) = P(0,0)/index;
            P(1,0) = P(1,0)/index;
            P(2,0) = P(2,0)/index;
            Q(0,0) = Q(0,0)/index;	
            Q(1,0) = Q(1,0)/index;
            Q(2,0) = Q(2,0)/index;

            Eigen::MatrixXd X(3,index);
            Eigen::MatrixXd Y(3,index);

            for(int i=0;i<index;i++){
                X(0,i) = pairs[i].p1.x - P(0,0);
                X(1,i) = pairs[i].p1.y - P(1,0);
                X(2,i) = pairs[i].p1.z - P(2,0);
                Y(0,i) = pairs[i].p2.x - Q(0,0);
                Y(1,i) = pairs[i].p2.y - Q(1,0);
                Y(2,i) = pairs[i].p2.z - Q(2,0);
            }
			//Perform Singular Value Decomposition on S
            Eigen::MatrixXd S = X * Y.transpose();
            JacobiSVD<MatrixXd> svd(S, ComputeFullV | ComputeFullU);
            Eigen::MatrixXd I ;
            I.setIdentity(svd.matrixV().cols(), svd.matrixU().cols());
            I(svd.matrixV().cols()-1,svd.matrixU().cols()-1) = (svd.matrixV() * svd.matrixU().transpose()).determinant();            
            Eigen::MatrixXd R = svd.matrixV() * I * svd.matrixU().transpose();            
            Eigen::MatrixXd t = Q - R*P;
            
            transformation_matrix(0,0) = R(0,0);
            transformation_matrix(0,1) = R(0,1);
            transformation_matrix(0,2) = R(0,2);
            transformation_matrix(1,0) = R(1,0);
            transformation_matrix(1,1) = R(1,1);
            transformation_matrix(1,2) = R(1,2);
            transformation_matrix(2,0) = R(2,0);
            transformation_matrix(2,1) = R(2,1);
            transformation_matrix(2,2) = R(2,2);
            transformation_matrix(0,3) = t(0,0);
            transformation_matrix(1,3) = t(1,0);
            transformation_matrix(2,3) = t(2,0);
      
            final_transformation_matrix *= transformation_matrix;           
    }

  	return final_transformation_matrix*initTransform;
}

/*
NDT from scratch
Eigen::Matrix4d NDT1(){
}

*/



int main(){
  
  	/*  **IMPORTANT**
    Parameters to set
    1.algroithm - To choose the algorithm to allign Localize the car - Can be ICP or NDT
    2.from_scratch - ICP algorithm from scratch has been implemented.Set this to true to call that function.*Not available for NDT yet*
    3.mapping - The lidar scan cloud is used to create a map of the surrounding if this flag is set to true. Press 's' to save the map.
    4.odometry - Uses a very basic odometer function to guess the next pose before it is sent to any allign algorithm.ICP and NDT algorithms converge must faster if this is used and the speed can be increased to upto 6 taps. 
    5.slam - Implements SLAM algorithm.*Set mapping to true*.Uses the map created from scans to localize instead of using the given Map.Simultaneous Localization and Mapping. 
    Default params:algorithm=NDT;from_scratch = false;mapping = false;odometry = true;slam = false;Set ndt iterations to 4
    Params for slam:algorithm=NDT;from_scratch = false;mapping = true;odometry = true;slam = true;Set ndt iterations to 8
    */
  	string algorithm = "NDT"; // CAN BE ICP OR NDT  	
  	bool from_scratch = false; //To call from scratch algorithms.Only available for ICP algorithm
  	bool mapping = false; //To cretae a map of the surroundings.*Uses computer resources.Can slow down the Simulation*
  	bool odometry = true;//Calls odometry function.Speeds up allignment.
  	bool slam = false; //To implement SLAM.*Uses computer resources.Can slow down the Simulation*
  	
  
  	
  	bool mapping_done = false;
  	bool first_map = true;
  
  	if(from_scratch){
    	algorithm += "1"; 	
    }

	auto client = cc::Client("localhost", 2000);
	client.SetTimeout(2s);
	auto world = client.GetWorld();

	auto blueprint_library = world.GetBlueprintLibrary();
	auto vehicles = blueprint_library->Filter("vehicle");

	auto map = world.GetMap();
	auto transform = map->GetRecommendedSpawnPoints()[1];
	auto ego_actor = world.SpawnActor((*vehicles)[12], transform);

	//Create lidar
	auto lidar_bp = *(blueprint_library->Find("sensor.lidar.ray_cast"));
	// CANDO: Can modify lidar values to get different scan resolutions
	lidar_bp.SetAttribute("upper_fov", "15");
    lidar_bp.SetAttribute("lower_fov", "-25");
    lidar_bp.SetAttribute("channels", "32");
    lidar_bp.SetAttribute("range", "30");
	lidar_bp.SetAttribute("rotation_frequency", "60");
	lidar_bp.SetAttribute("points_per_second", "500000");

	auto user_offset = cg::Location(0, 0, 0);
	auto lidar_transform = cg::Transform(cg::Location(-0.5, 0, 1.8) + user_offset);
	auto lidar_actor = world.SpawnActor(lidar_bp, lidar_transform, ego_actor.get());
	auto lidar = boost::static_pointer_cast<cc::Sensor>(lidar_actor);
	bool new_scan = true;
	std::chrono::time_point<std::chrono::system_clock> lastScanTime, startTime;

	pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  	viewer->setBackgroundColor (0, 0, 0);
	viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)&viewer);

	auto vehicle = boost::static_pointer_cast<cc::Vehicle>(ego_actor);
	Pose pose(Point(0,0,0), Rotate(0,0,0));

	// Load map
	PointCloudT::Ptr mapCloud1(new PointCloudT);
  	pcl::io::loadPCDFile("map.pcd", *mapCloud1);
  	//cout << "Loaded " << mapCloud->points.size() << " data points from map.pcd" << endl;
	renderPointCloud(viewer, mapCloud1, "map", Color(0,0,1)); 
  
  	typename pcl::PointCloud<PointT>::Ptr mapCloud (new pcl::PointCloud<PointT>);
    double res = 0.1;
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud( mapCloud1 );
    vg.setLeafSize(res, res, res);
    vg.filter(*mapCloud );
    cout << "Loaded " << mapCloud->points.size() << " data points from map.pcd" << endl;

	typename pcl::PointCloud<PointT>::Ptr cloudFiltered (new pcl::PointCloud<PointT>);
	typename pcl::PointCloud<PointT>::Ptr scanCloud (new pcl::PointCloud<PointT>);
	
  	
  	typename pcl::PointCloud<PointT>::Ptr slamMap (new pcl::PointCloud<PointT>);
  	
	lidar->Listen([&new_scan, &lastScanTime, &scanCloud](auto data){

		if(new_scan){
			auto scan = boost::static_pointer_cast<csd::LidarMeasurement>(data);
			for (auto detection : *scan){
				if((detection.point.x*detection.point.x + detection.point.y*detection.point.y + detection.point.z*detection.point.z) > 8.0){ // Don't include points touching ego
					pclCloud.points.push_back(PointT(detection.point.x, detection.point.y, detection.point.z));
				}
			}
			if(pclCloud.points.size() > 8000){ // CANDO: Can modify this value to get different scan resolutions
				lastScanTime = std::chrono::system_clock::now();
				*scanCloud = pclCloud;
				new_scan = false;
              	
			}
		}
	});
	
	Pose poseRef(Point(vehicle->GetTransform().location.x, vehicle->GetTransform().location.y, vehicle->GetTransform().location.z), Rotate(vehicle->GetTransform().rotation.yaw * pi/180, vehicle->GetTransform().rotation.pitch * pi/180, vehicle->GetTransform().rotation.roll * pi/180));
	double maxError = 0;
  	
  	Eigen::Matrix4d transform1 = transform3D(pose.rotation.yaw, pose.rotation.pitch, pose.rotation.roll, pose.position.x, pose.position.y, pose.position.z);
	PointCloudT::Ptr transformed_scan (new PointCloudT);
  	
  	Pose last_pose(Point(0,0,0), Rotate(0,0,0));
  	Pose lpose(Point(0,0,0), Rotate(0,0,0));
  	Pose llpose(Point(0,0,0), Rotate(0,0,0));
	
	//Initial Setup
  	pcl::KdTreeFLANN<PointT> kdtree;
  	pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;	
  	if(algorithm=="ICP1"){      
  	  kdtree.setInputCloud(mapCloud);
    }  
  	if(algorithm=="NDT"){            
      ndt.setTransformationEpsilon (.0001);  	
      ndt.setStepSize (1);  	
      ndt.setResolution (1);
      ndt.setInputTarget (mapCloud);   	  
  	}  
  	double total_error = 0.0;
  	double avg_error;
  	int count = 0;
  
	while (!viewer->wasStopped())
  	{
		while(new_scan){
			std::this_thread::sleep_for(0.1s);
			world.Tick(1s);
		}
		if(refresh_view){
			viewer->setCameraPosition(pose.position.x, pose.position.y, 60, pose.position.x+1, pose.position.y+1, 0, 0, 0, 1);
			refresh_view = false;
		}
		
		viewer->removeShape("box0");
		viewer->removeShape("boxFill0");
		Pose truePose = Pose(Point(vehicle->GetTransform().location.x, vehicle->GetTransform().location.y, vehicle->GetTransform().location.z), Rotate(vehicle->GetTransform().rotation.yaw * pi/180, vehicle->GetTransform().rotation.pitch * pi/180, vehicle->GetTransform().rotation.roll * pi/180)) - poseRef;
		drawCar(truePose, 0,  Color(1,0,0), 0.7, viewer);
		double theta = truePose.rotation.yaw;
		double stheta = control.steer * pi/4 + theta;
		viewer->removeShape("steer");
		renderRay(viewer, Point(truePose.position.x+2*cos(theta), truePose.position.y+2*sin(theta),truePose.position.z),  Point(truePose.position.x+4*cos(stheta), truePose.position.y+4*sin(stheta),truePose.position.z), "steer", Color(0,1,0));


		ControlState accuate(0, 0, 1);
		if(cs.size() > 0){
			accuate = cs.back();
			cs.clear();

			Accuate(accuate, control);
			vehicle->ApplyControl(control);
		}

  		viewer->spinOnce ();
		
		if(!new_scan){
			
			new_scan = true;
			//(Filter scan using voxel filter)
          	typename pcl::PointCloud<PointT>::Ptr cloudFiltered (new pcl::PointCloud<PointT>);
            double res = 0.5;
            pcl::VoxelGrid<PointT> vg;
            vg.setInputCloud( scanCloud );
            vg.setLeafSize(res, res, res);
            vg.filter(*cloudFiltered );
          
          	//Finding the transform between scan and given map.
          	if(algorithm=="ICP"){
              transform1 = ICP(mapCloud, cloudFiltered, pose, 8);
            }
          	else if(algorithm=="NDT"){
              transform1 = NDT(ndt, cloudFiltered, pose, 5);//5 for normal 8 for SLAM
            }
          	else if(algorithm=="ICP1"){
              transform1 = ICP1( mapCloud, cloudFiltered, pose, 30,kdtree);   
            }
          	else{
            	std::cout<<"Wrong set of parameters";
            	return 0;
            }

			//pose = ....
          	pose = getPose(transform1);

			//Transform scan so it aligns with ego's actual pose and render that scan
          	pcl::transformPointCloud (*cloudFiltered, *transformed_scan, transform1);  
          
			viewer->removePointCloud("scan");
			// TODO: Change `scanCloud` below to your transformed scan
			renderPointCloud(viewer, transformed_scan, "scan", Color(1,0,0) );
          	
          
          	viewer->removeAllShapes();
          	
         	//mapping 
          	//Points are added to map if the pose differnce exceeds 2m.Can be tweaked in the if statement
          	if(mapping && (getDistance(Point(pose.position.x, pose.position.y, pose.position.z), Point(last_pose.position.x, last_pose.position.y, last_pose.position.z))>2.0 || first_map)){
              	//uncomment for high res map.
                //pcl::transformPointCloud (*scanCloud, *transformed_scan, transform1);
                for(PointT p:transformed_scan->points){
                    slamMap->points.push_back(p);                  
                }
                last_pose = pose;
                first_map = false;
                viewer->removePointCloud("mapp");
                renderPointCloud(viewer, slamMap, "mapp", Color(0.5,0.5,0.5) );
              	//To save the map.
                //User has to press the s key or Pass the evaluation.
                if(save_map){
                  save_map = false;
                  PointCloudT::Ptr mapCloud(new PointCloudT);
                  *mapCloud = *slamMap;
                  mapCloud->width = mapCloud->points.size();
                  mapCloud->height = 1;
                  pcl::io::savePCDFileASCII ("my_map.pcd", *mapCloud);
                  cout << "saved pcd map" << endl;
                 }
              	mapping_done = true;              	
            }
			
            if(slam && mapping_done){
              	*mapCloud = *slamMap;//Using scanned map for Target Cloud insted of using ground truth
              	viewer->removePointCloud("map");
              	if(algorithm=="ICP1"){
                	kdtree.setInputCloud(mapCloud);
                }  
              	if(algorithm=="NDT"){
                	ndt.setInputTarget(mapCloud);
                }
            	mapping_done = false;
              	              	
            }
          	
          
			//viewer->removeAllShapes();
			drawCar(pose, 1,  Color(0,1,0), 0.35, viewer);
          	
          	count++;          
          	double poseError = sqrt( (truePose.position.x - pose.position.x) * (truePose.position.x - pose.position.x) + (truePose.position.y - pose.position.y) * (truePose.position.y - pose.position.y) );    
               	
          	//A very basic odometer algorithm.
          	//Assumes the distance travelled between Evry scan is constant.          
          	if(odometry){
              	llpose = pose;//Current calculated pose is saved to llpose
            	pose.position.x += pose.position.x - lpose.position.x;//pose is shifted by the differnce between last two poses.
              	pose.position.y += pose.position.y - lpose.position.y;
              	pose.position.z += pose.position.z - lpose.position.z;
              	lpose = llpose;//Current calculated pose is saved to lpose
              	viewer->removeShape("odo");
				viewer->addText("Using Odometry", 10, 250, 16, 0.0, 1.0, 1.0, "odo",0);              	
            }
          
			total_error +=poseError;
          	if(poseError > maxError)
				maxError = poseError;
			double distDriven = sqrt( (truePose.position.x) * (truePose.position.x) + (truePose.position.y) * (truePose.position.y) );
			viewer->removeShape("maxE");
			viewer->addText("Max Error: "+to_string(maxError)+" m", 200, 100, 32, 1.0, 1.0, 1.0, "maxE",0);
			viewer->removeShape("derror");
			viewer->addText("Pose error: "+to_string(poseError)+" m", 200, 150, 32, 1.0, 1.0, 1.0, "derror",0);
			viewer->removeShape("dist");
			viewer->addText("Distance: "+to_string(distDriven)+" m", 200, 200, 32, 1.0, 1.0, 1.0, "dist",0);
          	viewer->removeShape("throttle");
			viewer->addText("throttle: "+to_string(static_cast<int>(control.throttle*10))+"taps",500, 50, 16, 0.0, 1.0, 1.0, "throttle",0);
          	viewer->removeShape("steerr");
			viewer->addText("steer: "+to_string(static_cast<int>(control.steer*50))+"taps", 500, 25, 16, 0.0, 1.0, 1.0, "steerr",0);
          	viewer->removeShape("yaw");//Helps to steer the vehicle
			viewer->addText("yaw: "+to_string(static_cast<int>(vehicle->GetTransform().rotation.yaw))+"degree", 200, 50, 16, 0.0, 1.0, 1.0, "yaw",0);
          	avg_error = total_error/count;
          	viewer->removeShape("avg");//Average Pose error
			viewer->addText("averagePoseError: "+to_string(avg_error)+"m", 200, 25, 16, 0.0, 1.0, 1.0, "avg",0);
          	viewer->removeShape("alg");
			viewer->addText("Algorithm: "+algorithm, 10, 300, 16, 0.0, 1.0, 1.0, "alg",0);
          	if(slam){
            	viewer->removeShape("slam");
				viewer->addText("SLAM:ON", 300, 300, 16, 0.0, 1.0, 1.0, "slam",0);  
            }
			if(mapping){
              	viewer->removeShape("mapping");
				viewer->addText("Mapping: ON", 10, 275, 16, 0.0, 1.0, 1.0, "mapping",0);
              	viewer->removeShape("save");
				viewer->addText("Press s to save map", 10, 20, 12, 0.0, 1.0, 1.0, "save",0);            	
            }
			if(maxError > 1.2 || distDriven >= 170.0 ){
				viewer->removeShape("eval");
			if(maxError > 1.2){
				viewer->addText("Try Again", 200, 50, 32, 1.0, 0.0, 0.0, "eval",0);
			}
			else{
				viewer->addText("Passed!", 200, 50, 32, 0.0, 1.0, 0.0, "eval",0);
              	save_map = true;
			}
            	//viewer->saveScreenshot(algorithm+to_string(count)+".png");
		}
			
			pclCloud.points.clear();
		}
  	}
	return 0;
}
