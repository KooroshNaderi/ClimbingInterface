/*************************************************************************
 *                                                                       *
 * Climber's Interface done by Kourosh Naderi and Perttu Hamalainen      *
 * All rights reserved.  Email: firstname.lastname@aalto.fi              *
 *                                                                       *                  *
 *                                                                       *
 *************************************************************************/

#include <stdio.h>
#include <conio.h>
#include <chrono>
#include <list>
#include <unordered_map>
#include <queue>
#include <stdint.h>
#include "ode/ode.h"
#include "MathUtils.h"
#include <Math.h>

#include <Eigen/Geometry> 

#include "ControlPBP.h"
#include "UnityOde.h"
#include "RenderClient.h"
#include "RenderCommands.h"
#include "Debug.h"
#include "DynamicMinMax.h"
#include "CMAES.hpp"
#include "ClippedGaussianSampling.h"
#include "RecursiveTCBSpline.h"
#include "FileUtils.h"
//#include "WorkerThreadManager.h"

#undef max
#undef min

using namespace std::chrono;
using namespace std;
using namespace AaltoGames;
using namespace Eigen;

#ifdef _MSC_VER
#pragma warning(disable:4244 4305)  /* for VC++, no precision loss complaints */
#endif

/// select correct drawing functions 
#ifdef dDOUBLE
#define dsDrawBox dsDrawBoxD
#define dsDrawSphere dsDrawSphereD
#define dsDrawCylinder dsDrawCylinderD
#define dsDrawCapsule dsDrawCapsuleD
#endif
enum OptimizerTypes
{
	otCPBP=0,otCMAES
};
static const OptimizerTypes optimizerType = otCPBP;

float xyz[3] = {4.0,-5.0f,0.5f}; 
float lightXYZ[3] = {-10,-10,10};
float lookAt[3] = {0.75f,0,2.0};


static bool pause = false;
static bool useOfflinePlanning = optimizerType == otCMAES ? true : /*for C-PBP user can decide wheather do it online (false) or offline (true)*/ false;
static bool flag_capture_video = false;
static int maxNumSimulatedPaths = 3;

enum mCaseStudy{movingLimbs = 0, Energy = 1};
//case study: {path with minimum moving limbs, path with minimum energy} among "maxNumSimulatedPaths" paths founds
static int maxCaseStudySimulation = 2; 

static float mBiasWallY = -0.2f;
static float maxSpeed = optimizerType == otCMAES ? 2.0f*3.1415f : 2.0f*3.1415f;
static const float poseAngleSd=deg2rad*45.0f;
static const float maxSpeedRelToRange=4.0f; //joint with PI rotation range will at max rotate at this times PI radians per second
static float controlDiffSdScale =useOfflinePlanning ? 0.5f : 0.2f;
// for online method it works best if nPhysicsPerStep == 1. Also, Putting nPhysicsPerStep to 3 does not help in online method and makes simulation awful (bad posture, not reaching a stance)
static int nPhysicsPerStep = optimizerType == otCMAES ? 1 : (useOfflinePlanning ? 3 : 1);


static const float noCostImprovementThreshold = optimizerType == otCMAES ? 50.0f : 10.0f;
static const float cTime = 30.0f;
static float contactCFM = 0; 

//CMAES params
static const int nCMAESSegments = 3;
static const bool cmaesLinearInterpolation=true;
static const bool forceCMAESLastValueToZero=false;
static const float minSegmentDuration=0.25f;
static const float maxSegmentDuration=0.75f;
static const float maxFullTrajectoryDuration=3.0f;
static const float motorPoseInterpolationTime=0.0333f;
static const float torsoMinFMax=20.0f;
static const bool optimizeFMax=true;
static const bool cmaesSamplePoses=false;
static const bool useThreads=true;

enum FMaxGroups
{
	fmTorso=0,fmLeftArm ,fmLeftLeg,fmRightArm,fmRightLeg,
	fmEnd
};
static const int fmCount=optimizeFMax ? fmEnd : 0;


//Simulation globals
static float worldCfM = 1e-3;
static float worldERP = 1e-5;
static const float maximumForce = 350.0f;
static const float minimumForce = 2.0f; //non-zero FMax better for joint stability
static const float forceCostSd=optimizerType == otCMAES ? maximumForce : maximumForce;  //for computing control cost

static const bool testClimber=true;

enum mEnumTestCaseClimber
{
	TestAngle = 0, TestCntroller = 1
};

mEnumTestCaseClimber TestID = mEnumTestCaseClimber::TestCntroller;

// initial rotation (for cost computing in t-pose)
static	__declspec(align(16)) Eigen::Quaternionf initialRotations[100];  //static because the class of which this should be a member should be modified to support alignment...


 /// some constants 
static const float timeStep = 1.0f / cTime;   //physics simulation time step
#define BodyNUM 11+4		       // number of boxes 
//Note: CMAES uses this only for the first iteration, next ones use 1/4th of the samples
static const int contextNUM = optimizerType==otCMAES ? 257 : 65; //maximum number of contexts, //we're using 32 samples, i.e., forward-simulated trajectories per animation frame.
#define boneRadius (0.2f)	   // sphere radius 
#define boneLength (0.5f)	   // sphere radius
#define DENSITY 1.0f		// density of all objects
#define holdSize 0.5f

static const int nTrajectories = contextNUM - 1;
static int nTimeSteps = useOfflinePlanning ? int(cTime*1.5001f) : int(cTime/2);

#define Vector2 Eigen::Vector2f
#define Vector3 Eigen::Vector3f
#define Vector4 Eigen::Vector4f

#define DEG_TO_RAD (M_PI/180.0)

void start(); /// start simulation - set viewpoint 
void simLoop (int); /// simulation loop
void runSimulation(int, char **); /// run simulation

enum mDemoTestClimber{DemoRoute1 = 1, DemoRoute2 = 2, DemoRoute3 = 3, DemoLongWall = 4, Demo45Wall = 5};

static inline Eigen::Quaternionf ode2eigenq(ConstOdeQuaternion odeq)
{
	return Eigen::Quaternionf(odeq[0],odeq[1],odeq[2],odeq[3]); 
}
static inline void eigen2odeq(const Eigen::Quaternionf &q, OdeQuaternion out_odeq)
{
	out_odeq[0]=q.w();
	out_odeq[1]=q.x();
	out_odeq[2]=q.y();
	out_odeq[3]=q.z();
}

//timing
high_resolution_clock::time_point t1;
void startPerfCount()
{
	t1 = high_resolution_clock::now();
}
int getDurationMs()
{
	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	return (int)(time_span.count()*1000.0);
}

//the full state of a body
class BodyState
{
public:
	BodyState()
	{
		setAngle(Vector4(0.0f,0.0f,0.0f,0.0f));
		setPos(Vector3(0.0f,0.0f,0.0f));
		setVel(Vector3(0.0f,0.0f,0.0f));
		setAVel(Vector3(0.0f,0.0f,0.0f));
		boneSize = boneLength;
		bodyType = 0;
	}

	Vector4 getAngle()
	{
		return Vector4(angle[0], angle[1], angle[2], angle[3]);
	}

	Vector3 getPos()
	{
		return Vector3(pos[0], pos[1], pos[2]);
	}

	Vector3 getVel()
	{
		return Vector3(vel[0], vel[1], vel[2]);
	}

	Vector3 getAVel()
	{
		return Vector3(aVel[0], aVel[1], aVel[2]);
	}

	float getBoneSize()
	{
		return boneSize;
	}

	int getBodyType()
	{
		return bodyType;
	}

	void setAngle(Vector4& iAngle)
	{
		angle[0] = iAngle[0];
		angle[1] = iAngle[1];
		angle[2] = iAngle[2];
		angle[3] = iAngle[3];
		return;
	}

	void setPos(Vector3& iPos)
	{
		pos[0] = iPos[0];
		pos[1] = iPos[1];
		pos[2] = iPos[2];
		return;
	}

	void setVel(Vector3& iVel)
	{
		vel[0] = iVel[0];
		vel[1] = iVel[1];
		vel[2] = iVel[2];
		return;
	}

	void setAVel(Vector3& iAVel)
	{
		aVel[0] = iAVel[0];
		aVel[1] = iAVel[1];
		aVel[2] = iAVel[2];
		return;
	}

	void setBoneSize(float iBonSize)
	{
		boneSize = iBonSize;
	}

	void setBodyType(float iBodyType)
	{
		bodyType = iBodyType;
	}
	
private:
	float angle[4];
	float pos[3];
	float vel[3];
	float aVel[3];

	float boneSize;
	int bodyType;
};

class BipedState
{
public:
	enum BodyName{ BodyTrunk = 0, BodyLeftThigh = 1, BodyRightThigh = 2, BodyLeftShoulder = 3, BodyRightShoulder = 4
				 , BodyLeftLeg = 5, BodyRightLeg = 6, BodyLeftArm = 7, BodyRightArm = 8, BodyHead = 9, BodySpine = 10
				 , BodyLeftHand = 11, BodyRightHand = 12, BodyLeftFoot = 13, BodyRightFoot = 14
				 , BodyTrunkUpper = 15, BodyTrunkLower = 16, BodyHold = 17 };
	enum BodyType{BodyCapsule = 0, BodyBox = 1, BodySphere = 2};

	BipedState()
	{
		for (int i = 0; i < BodyNUM; i++)
		{
			bodyStates.push_back(BodyState());
		}
		forces = std::vector<Vector3>(4, Vector3(0,0,0));
	}

	BipedState getNewCopy(int freeSlot, int fromContextSlot)
	{
		BipedState c;

		c.bodyStates = bodyStates;
		c.hold_bodies_ids = hold_bodies_ids;
		c.saving_slot_state = freeSlot;

		saveOdeState(freeSlot, fromContextSlot);

		return c;
	}

	std::vector<BodyState> bodyStates;
	int saving_slot_state;
	std::vector<int> hold_bodies_ids; // left-leg, right-get, left-hand, right-hand
	std::vector<Vector3> forces;

	Vector3 getBodyDirectionZ(int i)
	{
		dMatrix3 R;
		Vector4 mAngle;
		if (i == BodyName::BodyTrunkLower)
			mAngle = bodyStates[BodyName::BodySpine].getAngle();
		else
			mAngle = bodyStates[i].getAngle();
		dReal mQ[]= {mAngle[0], mAngle[1], mAngle[2], mAngle[3]};
		dRfromQ(R, mQ);

		int targetDirection = 2; // alwayse in z direction
		int t_i = i;
		if (i == BodyName::BodyTrunkLower)
			t_i = BodyName::BodySpine;
		if (bodyStates[t_i].getBodyType() == BodyType::BodyBox)
		{
			if (i == BodyName::BodyLeftArm || i == BodyName::BodyLeftShoulder || i == BodyName::BodyRightArm || i == BodyName::BodyRightShoulder)
				targetDirection = 0; // unless one of these bones is wanted
		}
		Vector3 targetDirVector(R[4 * 0 + targetDirection], R[4 * 1 + targetDirection], R[4 * 2 + targetDirection]);
		if (i == BodyName::BodyLeftArm || i == BodyName::BodyLeftShoulder || i == BodyName::BodyLeftThigh || i == BodyName::BodyRightThigh 
			|| i == BodyName::BodyLeftLeg || i == BodyName::BodyRightLeg || i == BodyName::BodyTrunkLower || i == BodyName::BodyLeftHand
			|| i == BodyName::BodyLeftFoot || i == BodyName::BodyRightFoot)
		{
			targetDirVector = -targetDirVector;
		}

		return targetDirVector;
	}

	Vector3 getEndPointPosBones(int i)
	{
		/*switch (i)
		{
			case BodyName::BodyLeftArm:
				return bodyStates[BodyName::BodyLeftHand].getPos();
				break;
			case BodyName::BodyLeftLeg:
				return bodyStates[BodyName::BodyLeftFoot].getPos();
				break;
			case BodyName::BodyRightArm:
				return bodyStates[BodyName::BodyRightHand].getPos();
				break;
			case BodyName::BodyRightLeg:
				return bodyStates[BodyName::BodyRightFoot].getPos();
				break;
		default:
			break;
		}*/

		switch (i)
		{
			case BodyName::BodyLeftArm:
				i = BodyName::BodyLeftHand;
				break;
			case BodyName::BodyLeftLeg:
				i = BodyName::BodyLeftFoot;
				break;
			case BodyName::BodyRightArm:
				i = BodyName::BodyRightHand;
				break;
			case BodyName::BodyRightLeg:
				i = BodyName::BodyRightFoot;
				break;
		default:
			break;
		}

		if (i == BodyName::BodyTrunk)
			i = BodyName::BodySpine;
		if (i == BodyName::BodyTrunkUpper)
			i = BodyName::BodyTrunk;
		
		Vector3 targetDirVector = getBodyDirectionZ(i);

		if (i == BodyName::BodyTrunkLower)
			i = BodyName::BodySpine;

		Vector3 mPos = bodyStates[i].getPos();

		float bone_size = bodyStates[i].getBoneSize() / 2;
		Vector3 ePos_i(mPos[0] + targetDirVector.x() * bone_size, mPos[1] + targetDirVector.y() * bone_size, mPos[2] + targetDirVector.z() * bone_size);

		return ePos_i;
	}
};

class mFileHandler
{
private:
	std::string mfilename;
	bool mExists;
	FILE * rwFileStream;

	bool mOpenCreateFile(const std::string& name) 
	{
		fopen_s(&rwFileStream, name.c_str(), "r");
		if (rwFileStream == nullptr) 
		{
			fopen_s(&rwFileStream, name.c_str(), "w+");
		}
		return true;
	}

	template<typename Out>
	void split(const std::string &s, char delim, Out result)
	{
		std::stringstream ss;
		ss.str(s);
		std::string item;
		while (std::getline(ss, item, delim)) 
		{
			*(result++) = item;
		}
	}

public:

	std::vector<std::string> split(const std::string &s, char delim)
	{
		std::vector<std::string> elems;
		split(s, delim, std::back_inserter(elems));
		return elems;
	}

	void mCloseFile()
	{
		if (mExists)
		{
			fclose(rwFileStream);
			mExists = false;
		}
	}

	bool reWriteFile(std::vector<float>& iValues)
	{
		mCloseFile();
		
		fopen_s(&rwFileStream , mfilename.c_str(), "w");
		fprintf(rwFileStream, "#val \n");
		for (unsigned int i = 0; i < iValues.size(); i++)
		{
			fprintf(rwFileStream, "%f \n", iValues[i]);
		}
		mExists = true;
		return true;
	}

	bool openFileForWritingOn()
	{
		mCloseFile();
		
		fopen_s(&rwFileStream , mfilename.c_str(), "w");

		return true;
	}

	void writeLine(std::string _str)
	{
		fputs(_str.c_str(), rwFileStream);
		return;
	}

	void readFile(std::vector<std::vector<float>>& values)
	{
		if (!mExists)
		{
			return;
		}

		int lineNum = 0;
		int found_comma = 0;
		while (!feof(rwFileStream))
		{
			if (lineNum == 0)
			{
				char buff[200];
				char* mLine = fgets(buff, 200, rwFileStream);
				if (mLine)
				{
					if (lineNum == 0)
					{
						for (unsigned int c = 0; c < strlen(mLine); c++)
						{
							if (mLine[c] == ',') found_comma++;
						}
					}
				}
			}
			else
			{
				std::vector<float> val(found_comma + 1, -1);
				char buff[200];
				// how to parse the file
				switch (found_comma)
				{
				case 0:
					if (fscanf_s(rwFileStream, "%f", &val[0]) > 0)
						values.push_back(val); // stof(buff)
					break;
				case 1:
					if (fscanf_s(rwFileStream, "%f,%f", &val[0], &val[1]) > 0)
						values.push_back(val);
					break;
				case 2:
					if (fscanf_s(rwFileStream, "%f,%f,%f", &val[0], &val[1], &val[2]) > 0)
						values.push_back(val);
					break;
				case 3:
					if (fscanf_s(rwFileStream, "%f,%f,%f,%f", &val[0], &val[1], &val[2], &val[3]) > 0)
						values.push_back(val);
					break;
				//case 4:
				//	if (fscanf_s(rwFileStream, "%f,%f,%f,%f,%f", &val[0], &val[1], &val[2], &val[3], &val[4]) > 0)
				//		values.push_back(val);
				//	break;
				//case 5:
				//	if (fscanf_s(rwFileStream, "%f,%f,%f,%f,%f,%f", &val[0], &val[1], &val[2], &val[3], &val[4], &val[5]) > 0)
				//		values.push_back(val);
				//	break;
				//case 6:
				//	if (fscanf_s(rwFileStream, "%f,%f,%f,%f,%f,%f,%f", &val[0], &val[1], &val[2], &val[3], &val[4], &val[5], &val[6]) > 0)
				//		values.push_back(val);
				//	break;
				//case 7:
				//	if (fscanf_s(rwFileStream, "%f,%f,%f,%f,%f,%f,%f,%f", &val[0], &val[1], &val[2], &val[3], &val[4], &val[5], &val[6], &val[7]) > 0)
				//		values.push_back(val);
				//	break;
				//case 8:
				//	if (fscanf_s(rwFileStream, "%f,%f,%f,%f,%f,%f,%f,%f,%f", &val[0], &val[1], &val[2], &val[3], &val[4], &val[5], &val[6], &val[7], &val[8]) > 0)
				//		values.push_back(val);
				//	break;
				default:
					if (fgets(buff, 200, rwFileStream) != NULL)
					{
						std::vector<std::string> _STR =	split(buff, ',');
						if (_STR.size() == val.size())
						{
							for (unsigned int i = 0; i < val.size(); i++)
							{
								val[i] = (float)strtod(_STR[i].c_str(), NULL);
							}
							values.push_back(val);
						}
					}
					break;
				}	
			}
			
			lineNum++;
		}
		return;
	}

	mFileHandler(const std::string& iFilename)
	{
		mfilename = iFilename;
		mExists = mOpenCreateFile(iFilename);
	}

	~mFileHandler()
	{
		mCloseFile();
	}
};

class SimulationContext
{
public:
	class holdContent{
	public:
		holdContent(Vector3 _holdPos, float _f_ideal, Vector3 _d_ideal, float _k, float _size, int _geomID, int _HoldPushPullMode) // Vector3 _notTransformedHoldPos
		{
			holdPos = _holdPos;
			f_ideal = _f_ideal;

			d_ideal = Vector3(_d_ideal[0], _d_ideal[1], _d_ideal[2]);
			if (d_ideal.norm() > 0)
				d_ideal.normalize();

			k = _k;
			size = _size;

			theta = atan2f(d_ideal[1], d_ideal[0]) * (180/PI);
			phi = asinf(d_ideal[2]) * (180/PI);
			geomID = _geomID;

	//		notTransformedHoldPos = _notTransformedHoldPos;

			HoldPushPullMode = _HoldPushPullMode;
		}
		Vector3 holdPos;
		float f_ideal;
		Vector3 d_ideal;
		float k;
		float size;
		int HoldPushPullMode;// 0 = pulling, 1 = pushing, 2 = pull/pushing

//		Vector3 notTransformedHoldPos;

		std::string toString(float _width)
		{
			//#x,y,z,f,dx,dy,dz,k,s
			std::string write_buff;
			char _buff[100];

			//cPos[0] = -wWidth / 2 + mHoldInfo[i][0];
			//cPos[1] = -(mHoldInfo[i][8] / 4) + mHoldInfo[i][1];
			//cPos[2] = mHoldInfo[i][2];

			sprintf_s(_buff, "%f,", holdPos[0] + _width / 2); write_buff += _buff; // notTransformedHoldPos[0]
			sprintf_s(_buff, "%f,", holdPos[1] + size / 4); write_buff += _buff; // notTransformedHoldPos[1]
			sprintf_s(_buff, "%f,", holdPos[2]); write_buff += _buff; // notTransformedHoldPos[2]

			sprintf_s(_buff, "%f,", f_ideal); write_buff += _buff;

			sprintf_s(_buff, "%f,", d_ideal[0]); write_buff += _buff;
			sprintf_s(_buff, "%f,", d_ideal[1]); write_buff += _buff;
			sprintf_s(_buff, "%f,", d_ideal[2]); write_buff += _buff;

			sprintf_s(_buff, "%f,", k); write_buff += _buff;

			sprintf_s(_buff, "%f,", size); write_buff += _buff;

			sprintf_s(_buff, "%d\n", HoldPushPullMode); write_buff += _buff;
			return write_buff;
		}

		// for updating the hold directions
		float theta;
		float phi;
		int geomID;
	};

	enum MouseTrackingBody{MouseLeftLeg = 0, MouseRightLeg = 1, MouseLeftHand = 2, MouseRightHand = 3, MouseTorsoDir = 4, MouseNone = -1};

	enum JointType{ FixedJoint = 0, OneDegreeJoint = 1, TwoDegreeJoint = 2, ThreeDegreeJoint = 3, BallJoint = 4};
	enum BodyType{BodyCapsule = 0, BodyBox = 1, BodySphere = 2};

	enum ContactPoints{ LeftLeg = 5, RightLeg = 6, LeftArm = 7, RightArm = 8 };
	enum BodyName{ BodyTrunk = 0, BodyLeftThigh = 1, BodyRightThigh = 2, BodyLeftShoulder = 3, BodyRightShoulder = 4
				 , BodyLeftLeg = 5, BodyRightLeg = 6, BodyLeftArm = 7, BodyRightArm = 8, BodyHead = 9, BodySpine = 10
				 , BodyLeftFoot = 11, BodyRightFoot = 12, BodyLeftHand = 13, BodyRightHand = 14
				 , BodyTrunkUpper = 15, BodyTrunkLower = 16, BodyHold = 17 };

	Vector3 initialPosition[contextNUM - 1];
	Vector3 resultPosition[contextNUM - 1];
	std::vector<holdContent> holds_body;

	mFileHandler mDesiredAngleFile;
	mFileHandler mClimberInfo;
	mFileHandler mClimberKinectInfo;
	mFileHandler mWallInfo;
	mFileHandler mHoldsInfo;

	int _RouteNum;
	float _wWidth, _wHeight;
	bool isTestClimber;

	~SimulationContext()
	{
		//mDesiredAngleFile.reWriteFile(desiredAnglesBones);
		//mDesiredAngleFile.mCloseFile();
		////#x,y,z,f,dx,dy,dz,k,s"
		//mFileHandler _holdFile(getAppendAddress("ClimberInfo\\mHoldsRoute", _RouteNum, ".txt"));
		//_holdFile.openFileForWritingOn();
		//_holdFile.writeLine("#x,y,z,f,dx,dy,dz,k,s,m\n");
		//for (unsigned int i = 0; i < holds_body.size(); i++)
		//{
		//	_holdFile.writeLine(holds_body[i].toString(_wWidth));
		//}
		//_holdFile.mCloseFile();
	}

	char mReadBuff[100];
	char* getAppendAddress(char* firstText, int num, char* secondText)
	{
		
		sprintf_s(&mReadBuff[0], 100, "%s%d%s", firstText, num, secondText);

		return &mReadBuff[0];
	}

	int readRouteNumFromFile()
	{
		mFileHandler mRouteNumFile("ClimberInfo\\mRouteNumFile.txt");
		std::vector<std::vector<float>> readFileRouteInfo;
		mRouteNumFile.readFile(readFileRouteInfo);
		_RouteNum = (int)readFileRouteInfo[0][0];
		mRouteNumFile.mCloseFile();
		return _RouteNum;
	}

	SimulationContext(bool testClimber, mEnumTestCaseClimber TestID, mDemoTestClimber DemoID)
		 :mDesiredAngleFile("ClimberInfo\\mDesiredAngleFile.txt"),
		 mClimberInfo(getAppendAddress("ClimberInfo\\mClimberInfoFile", _RouteNum = readRouteNumFromFile(), ".txt")),
		 mClimberKinectInfo("ClimberInfo\\mClimberReadKinect.txt"),
		 mWallInfo("ClimberInfo\\mWallInfoFile.txt"),
		 mHoldsInfo(getAppendAddress("ClimberInfo\\mHoldsRoute", _RouteNum = readRouteNumFromFile(), ".txt"))
	{
		isTestClimber = testClimber;

		bodyIDs = std::vector<int>(BodyNUM);
		mGeomID = std::vector<int>(BodyNUM); // for drawing stuff

		bodyTypes = std::vector<int>(BodyNUM);
		boneSize = std::vector<float>(BodyNUM);
		fatherBodyIDs = std::vector<int>(BodyNUM);
		jointIDs = std::vector<int>(BodyNUM - 1, -1);
		jointTypes = std::vector<int>(BodyNUM - 1);

		jointHoldBallIDs = std::vector<int>(4,-1);
		holdPosIndex = std::vector<std::vector<int>>(contextNUM, std::vector<int>(4,-1));

		maxNumContexts = contextNUM; 

		currentFreeSavingStateSlot = 0;

		//Allocate one simulation context for each sample, plus one additional "master" context
		initOde(maxNumContexts);  // contactgroup = dJointGroupCreate (1000000); //might be needed, right now it is 0
		setCurrentOdeContext(ALLTHREADS);
		odeRandSetSeed(0);
		odeSetContactSoftCFM(contactCFM);
		
		odeWorldSetCFM(worldCfM);
		odeWorldSetERP(worldERP);

		odeWorldSetGravity(0, 0, -9.81f);

		int spaceID = odeCreatePlane(0,0,0,1,0);

		std::vector<std::vector<float>> readFileWallInfo;
		mWallInfo.readFile(readFileWallInfo);
		float wWidth = 10.0f, wHeight = 15.0f;
		if (readFileWallInfo.size() > 0)
		{
			wWidth = readFileWallInfo[0][0];
			wHeight = readFileWallInfo[0][1];
		}
		mWallInfo.mCloseFile();

		std::vector<std::vector<float>> readFileClimberInfo;
		mClimberInfo.readFile(readFileClimberInfo);
		if (readFileClimberInfo.size() == 0)
		{
			std::vector<float> xyh;
			xyh.push_back(0.925f);
			xyh.push_back(0.5f);
			xyh.push_back(1.84f);
			readFileClimberInfo.push_back(xyh);
		}

		mClimberInfo.mCloseFile();
		
		std::vector<std::vector<float>> readFileClimberKinectInfo;
		mClimberKinectInfo.readFile(readFileClimberKinectInfo);
		mClimberKinectInfo.mCloseFile();

		createHumanoidBody(-wWidth / 2 + readFileClimberInfo[0][0] /*relative dis x with wall (m)*/,
						   -readFileClimberInfo[0][1] /*relative dis y with wall (m)*/, 
						   readFileClimberInfo[0][2] /*height of climber (m)*/, 
						   70.0f /*climber's mass (kg)*/,
						   readFileClimberKinectInfo);

		// calculate joint size
		mJointSize = 0;
		desiredAnglesBones.clear();
		for (int i = 0; i < BodyNUM - 1; i++)
		{
			mJointSize += jointTypes[i];
			for (int j = 0; j < jointTypes[i]; j++)
			{
				desiredAnglesBones.push_back(0.0f);
			}
		}

		std::vector<std::vector<float>> readFileHoldsInfo;
		mHoldsInfo.readFile(readFileHoldsInfo);
		mHoldsInfo.mCloseFile();

		if (!testClimber)
		{
			createEnvironment(DemoID, wWidth, wHeight, readFileHoldsInfo);

			attachContactPointToHold(ContactPoints::LeftArm, 2, ALLTHREADS);
			attachContactPointToHold(ContactPoints::RightArm, 3, ALLTHREADS);
			attachContactPointToHold(ContactPoints::LeftLeg, 0, ALLTHREADS);
			attachContactPointToHold(ContactPoints::RightLeg, 1, ALLTHREADS);
		}
		else
		{
			Vector3 hPos;
			switch (TestID)
			{
			case mEnumTestCaseClimber::TestAngle:
				hPos = getEndPointPosBones(SimulationContext::BodyName::BodyTrunk);
				createJointType(hPos.x(), hPos.y(), hPos.z(), -1, SimulationContext::BodyName::BodyTrunk, JointType::FixedJoint);
				break;
			case mEnumTestCaseClimber::TestCntroller:
				createEnvironment(DemoID, wWidth, wHeight, readFileHoldsInfo);

				attachContactPointToHold(ContactPoints::LeftArm, 2, ALLTHREADS);
				attachContactPointToHold(ContactPoints::RightArm, 3, ALLTHREADS);
				attachContactPointToHold(ContactPoints::LeftLeg, 0, ALLTHREADS);
				attachContactPointToHold(ContactPoints::RightLeg, 1, ALLTHREADS);
				break;
			}
		}

		// read desired angles from file
		std::vector<std::vector<float>> readFiledAngles;
		mDesiredAngleFile.readFile(readFiledAngles);
		if (readFiledAngles.size() > 0)
		{
			for (unsigned int i = 0; i < readFiledAngles.size(); i++)
			{
				desiredAnglesBones[i] = readFiledAngles[i][0];
			}
		}
		if (TestID != mEnumTestCaseClimber::TestAngle)
		{
			mDesiredAngleFile.mCloseFile();
		}

		for (int i = 0; i < maxNumContexts; i++)
		{
			int cContextSavingSlotNum = getNextFreeSavingSlot();
			saveOdeState(cContextSavingSlotNum);
		}

		//We're done, now we should have nSamples+1 copies of a model
		masterContext = contextNUM - 1;
		setCurrentOdeContext(masterContext);
	}

	static float getAbsAngleBtwVectors(Vector3 v0, Vector3 v1)
	{
		if (v0.norm() > 0)
			v0.normalize();
		if (v1.norm() > 0)
			v1.normalize();

		float _dotProduct = v0.x() * v1.x() + v0.y() * v1.y() + v0.z() * v1.z();
		float angle = 0.0f;
		if (fabs(_dotProduct) < 1)
			angle = acosf(_dotProduct);

		return angle;
	}

	static float getAngleBtwVectorsXZ(Vector3 _v0, Vector3 _v1)
	{
		Vector2 v0(_v0.x(), _v0.z());
		Vector2 v1(_v1.x(), _v1.z());

		v0.normalize();
		v1.normalize();

		float angle = acosf(v0.x() * v1.x() + v0.y() * v1.y());

		float crossproduct = (v0.x() * v1.y() - v0.y() * v1.x()) * 1;// in direction of k

		if (crossproduct < 0)
			angle = -angle;

		return angle;
	}

	void detachContactPoint(ContactPoints iEndPos, int targetContext)
	{
		int cContextNum = getCurrentOdeContext();

		setCurrentOdeContext(targetContext);

		int jointIndex = iEndPos - ContactPoints::LeftLeg;

		odeJointAttach(jointHoldBallIDs[jointIndex], 0, 0);

		//if (mENVGeoms.size() > 0)
		//{
		//	unsigned long cCollideBits = odeGeomGetCollideBits(mENVGeoms[0]); // collision bits of wall
		//	switch (jointIndex)
		//	{
		//	case 0: // left leg should collide with 0x0040
		//		odeGeomSetCollideBits(mENVGeoms[0], cCollideBits | unsigned long(0x0040));
		//		break;
		//	case 1: // right leg should collide with 0x0008
		//		odeGeomSetCollideBits(mENVGeoms[0], cCollideBits | unsigned long(0x0008));
		//		break;
		//	case 2: // left hand should collide with 0x2000
		//		odeGeomSetCollideBits(mENVGeoms[0], cCollideBits | unsigned long(0x2000));
		//		break;
		//	case 3: // right hand should collide with 0x0400
		//		odeGeomSetCollideBits(mENVGeoms[0], cCollideBits | unsigned long(0x0400));
		//		break;
		//	default:
		//		break;
		//	}
		//}

		if (targetContext == ALLTHREADS)
		{
			for (int i = 0; i < maxNumContexts; i++)
			{
				holdPosIndex[i][jointIndex] = -1;
			}
		}
		else
			holdPosIndex[targetContext][jointIndex] = -1;

		setCurrentOdeContext(cContextNum);
	}

	void attachContactPointToHold(ContactPoints iEndPos, int iHoldID, int targetContext)
	{
		int cContextNum = getCurrentOdeContext();

		setCurrentOdeContext(targetContext);

		int jointIndex = iEndPos - ContactPoints::LeftLeg;
		int boneIndex = BodyName::BodyLeftLeg + jointIndex;
		Vector3 hPos = getEndPointPosBones(boneIndex);

		switch (boneIndex)
		{
			case BodyName::BodyLeftArm:
				boneIndex = BodyName::BodyLeftHand;
				break;
			case BodyName::BodyLeftLeg:
				boneIndex = BodyName::BodyLeftFoot;
				break;
			case BodyName::BodyRightArm:
				boneIndex = BodyName::BodyRightHand;
				break;
			case BodyName::BodyRightLeg:
				boneIndex = BodyName::BodyRightFoot;
				break;
		default:
			break;
		}

		bool flag_set_holdIndex = false;
		if (jointHoldBallIDs[jointIndex] == -1) // create the hold joint only once
		{
			int cJointBallID = createJointType(hPos.x(), hPos.y(), hPos.z(), -1, boneIndex);

			jointHoldBallIDs[jointIndex] = cJointBallID;
			flag_set_holdIndex = true;
		}

		Vector3 holdPos = getHoldPos(iHoldID);
		float _dis = (holdPos - hPos).norm();

		float _connectionThreshold = 0.24f * getHoldSize(iHoldID);

		if (_dis <= _connectionThreshold + 0.1f)
		{
			if (jointHoldBallIDs[jointIndex] != -1)
			{
				odeJointAttach(jointHoldBallIDs[jointIndex], 0, bodyIDs[boneIndex]);
				odeJointSetBallAnchor(jointHoldBallIDs[jointIndex], hPos.x(), hPos.y(), hPos.z());
				flag_set_holdIndex = true;
			}
		}
		
		if (flag_set_holdIndex)
		{
			//if (mENVGeoms.size() > 0)
			//{
			//	unsigned long cCollideBits = odeGeomGetCollideBits(mENVGeoms[0]); // collision bits of wall
			//	switch (jointIndex)
			//	{
			//	case 0: // left leg should not collide with 0x0040
			//		odeGeomSetCollideBits(mENVGeoms[0], cCollideBits & unsigned long(0xFFBF));
			//		break;
			//	case 1: // right leg should not collide with 0x0008
			//		odeGeomSetCollideBits(mENVGeoms[0], cCollideBits & unsigned long(0xFFF7));
			//		break;
			//	case 2: // left hand should not collide with 0x2000
			//		odeGeomSetCollideBits(mENVGeoms[0], cCollideBits & unsigned long(0xDFFF));
			//		break;
			//	case 3: // right hand should not collide with 0x0400
			//		odeGeomSetCollideBits(mENVGeoms[0], cCollideBits & unsigned long(0xFBFF));
			//		break;
			//	default:
			//		break;
			//	}
			//}

			if (targetContext == ALLTHREADS)
			{
				for (int i = 0; i < maxNumContexts; i++)
				{
					holdPosIndex[i][jointIndex] = iHoldID;
				}
			}
			else
				holdPosIndex[targetContext][jointIndex] = iHoldID;
		}

		setCurrentOdeContext(cContextNum);

		return;
	}

	int getNextFreeSavingSlot() 
	{
		return currentFreeSavingStateSlot++;
	}

	int getMasterContextID()
	{
		return masterContext;
	}

	////////////////////////////////////////// get humanoid and hold bodies info /////////////////////////

	Vector3 getHoldPos(int i)
	{
		return holds_body[i].holdPos;
	}

	float getHoldSize(int i)
	{
		return holds_body[i].size;
	}

	int getIndexHoldFromGeom(int cGeomID)
	{
		for (unsigned int i = 0; i < mENVGeoms.size(); i++)
		{
			if (mENVGeoms[i] == cGeomID)
			{
				return i - startingHoldGeomsIndex;
			}
		}
		return -1;
	}

	int getIndexHold(int cGeomID)
	{
		for (unsigned int i = 0; i < holds_body.size(); i++)
		{
			if (holds_body[i].geomID == cGeomID)
			{
				return i;
			}
		}
		return -1;
	}

	MouseTrackingBody getIndexHandsAndLegsFromGeom(int cGeomID)
	{
		if (cGeomID == -1)
			return MouseNone;
		for (int i = 0; i < 5; i++)
		{
			switch (i)
			{
			case 0:
				if (mGeomID[BodyLeftFoot] == cGeomID)
					return MouseLeftLeg;
				break;
			case 1:
				if (mGeomID[BodyRightFoot] == cGeomID)
					return MouseRightLeg;
				break;
			case 2:
				if (mGeomID[BodyLeftHand] == cGeomID)
					return MouseLeftHand;
				break;
			case 3:
				if (mGeomID[BodyRightHand] == cGeomID)
					return MouseRightHand;
				break;
			case 4:
				if (mGeomID[BodyTrunk] == cGeomID)
					return MouseTorsoDir;
				break;
			default:
				break;
			}
		}
		return MouseNone;
	}

	bool checkViolatingRelativeDis()
	{
		for (unsigned int i = 0; i < fatherBodyIDs.size(); i++)
		{
			if (fatherBodyIDs[i] != -1)
			{
				Vector3 bone_i = getEndPointPosBones(i, true);
				Vector3 f_bone_i = getEndPointPosBones(fatherBodyIDs[i], true);

				float coeff = 1.5f;

				if (fatherBodyIDs[i] == BodyName::BodyTrunkUpper)
					coeff = 2.0f;

				float cDis = (bone_i - f_bone_i).norm();
				float threshold = coeff * boneSize[i] + 0.1f;
				if (cDis > threshold)
				{
					return true;
				}
			}
		}

		return false;
	}

	Vector3 getBodyDirectionZ(int i)
	{
		dMatrix3 R;
		ConstOdeQuaternion mQ;
		if (i == BodyName::BodyTrunkLower)
			mQ = odeBodyGetQuaternion(bodyIDs[BodyName::BodySpine]);
		else
			mQ = odeBodyGetQuaternion(bodyIDs[i]);
		dRfromQ(R, mQ);

		int targetDirection = 2; // alwayse in z direction
		int t_i = i;
		if (i == BodyName::BodyTrunkLower)
			t_i = BodyName::BodySpine;
		if (bodyTypes[t_i] == BodyType::BodyBox)
		{
			if (i == BodyName::BodyLeftArm || i == BodyName::BodyLeftShoulder || i == BodyName::BodyRightArm || i == BodyName::BodyRightShoulder)
				targetDirection = 0; // unless one of these bones is wanted
		}
		Vector3 targetDirVector(R[4 * 0 + targetDirection], R[4 * 1 + targetDirection], R[4 * 2 + targetDirection]);
		if (i == BodyName::BodyLeftArm || i == BodyName::BodyLeftShoulder || i == BodyName::BodyLeftThigh || i == BodyName::BodyRightThigh 
			|| i == BodyName::BodyLeftLeg || i == BodyName::BodyRightLeg || i == BodyName::BodyTrunkLower || i == BodyName::BodyLeftHand
			|| i == BodyName::BodyLeftFoot || i == BodyName::BodyRightFoot)
		{
			targetDirVector = -targetDirVector;
		}

		return targetDirVector;
	}

	int getHoldBodyIDs(int i, int targetContext)
	{
		if (targetContext == ALLTHREADS)
			return holdPosIndex[maxNumContexts-1][i];
		else
			return holdPosIndex[targetContext][i];
	}

	int getHoldBodyIDsSize()
	{
		return holdPosIndex[0].size(); // the value is 4
	}

	int getJointSize()
	{
		return mJointSize;
	}

	Vector3 getBonePosition(int i)
	{
		ConstOdeVector rPos;
		
		if (i < BodyNUM)
			rPos = odeBodyGetPosition(bodyIDs[i]);
		else
		{
			//rPos = odeBodyGetPosition(bodyHoldIDs[i - BodyNUM]);
			return Vector3(0.0f,0.0f,0.0f);
		}
		
		return Vector3(rPos[0], rPos[1], rPos[2]);
	}
	
	Vector3 getBoneLinearVelocity(int i)
	{
		ConstOdeVector rVel = odeBodyGetLinearVel(bodyIDs[i]);
		
		return Vector3(rVel[0], rVel[1], rVel[2]);
	}

	Vector4 getBoneAngle(int i)
	{
		ConstOdeQuaternion rAngle = odeBodyGetQuaternion(bodyIDs[i]);

		return Vector4(rAngle[0], rAngle[1], rAngle[2], rAngle[3]);
	}

	Vector3 getBoneAngularVelocity(int i)
	{
		ConstOdeVector rAVel = odeBodyGetAngularVel(bodyIDs[i]);
		
		return Vector3(rAVel[0], rAVel[1], rAVel[2]);
	}

	float getJointAngle(int i)
	{
		if (jointTypes[jointIDIndex[i]] == JointType::ThreeDegreeJoint)
		{
			float angle = odeJointGetAMotorAngle(jointIDs[jointIDIndex[i]], jointAxisIndex[i]);
			return angle;
		}
		return odeJointGetHingeAngle(jointIDs[jointIDIndex[i]]);
	}
	
	float getJointAngleRate(int i)
	{
		if (jointTypes[jointIDIndex[i]] == JointType::ThreeDegreeJoint)
		{
			float angle = odeJointGetAMotorAngleRate(jointIDs[jointIDIndex[i]], jointAxisIndex[i]);
			return angle;
		}
		return odeJointGetHingeAngleRate(jointIDs[jointIDIndex[i]]);
	}
	
	float getJointFMax(int i)
	{
		if (jointTypes[jointIDIndex[i]] == JointType::ThreeDegreeJoint)
		{
			return odeJointGetAMotorParam(jointIDs[jointIDIndex[i]], dParamFMax1 + dParamGroup*jointAxisIndex[i]);
		}
		return odeJointGetHingeParam(jointIDs[jointIDIndex[i]], dParamFMax1 );
	}

	float getJointAngleMin(int i)
	{
		if (jointTypes[jointIDIndex[i]] == JointType::ThreeDegreeJoint)
		{
			return odeJointGetAMotorParam(jointIDs[jointIDIndex[i]], dParamLoStop1 + dParamGroup*jointAxisIndex[i]);
		}
		return odeJointGetHingeParam(jointIDs[jointIDIndex[i]], dParamLoStop1 );
	}
	
	float getJointAngleMax(int i)
	{
		if (jointTypes[jointIDIndex[i]] == JointType::ThreeDegreeJoint)
		{
			return odeJointGetAMotorParam(jointIDs[jointIDIndex[i]], dParamHiStop1 + dParamGroup*jointAxisIndex[i]);
		}
		return odeJointGetHingeParam(jointIDs[jointIDIndex[i]], dParamHiStop1 );
	}

	Vector3 getEndPointPosBones(int i, bool flag_calculate_exact_val = false)
	{
		ConstOdeVector mPos;
		/*switch (i)
		{
			case BodyName::BodyLeftArm:
				mPos = odeBodyGetPosition(bodyIDs[BodyName::BodyLeftHand]);
				return Vector3(mPos[0], mPos[1], mPos[2]);
				break;
			case BodyName::BodyLeftLeg:
				mPos = odeBodyGetPosition(bodyIDs[BodyName::BodyLeftFoot]);
				return Vector3(mPos[0], mPos[1], mPos[2]);
				break;
			case BodyName::BodyRightArm:
				mPos = odeBodyGetPosition(bodyIDs[BodyName::BodyRightHand]);
				return Vector3(mPos[0], mPos[1], mPos[2]);
				break;
			case BodyName::BodyRightLeg:
				mPos = odeBodyGetPosition(bodyIDs[BodyName::BodyRightFoot]);
				return Vector3(mPos[0], mPos[1], mPos[2]);
				break;
		default:
			break;
		}*/
		if (!flag_calculate_exact_val)
		{
			switch (i)
			{
				case BodyName::BodyLeftArm:
					i = BodyName::BodyLeftHand;
					break;
				case BodyName::BodyLeftLeg:
					i = BodyName::BodyLeftFoot;
					break;
				case BodyName::BodyRightArm:
					i = BodyName::BodyRightHand;
					break;
				case BodyName::BodyRightLeg:
					i = BodyName::BodyRightFoot;
					break;
			default:
				break;
			}
		}
		if (i == BodyName::BodyTrunk)
			i = BodyName::BodySpine;
		if (i == BodyName::BodyTrunkUpper)
			i = BodyName::BodyTrunk;
		
		Vector3 targetDirVector = getBodyDirectionZ(i);

		if (i == BodyName::BodyTrunkLower)
			i = BodyName::BodySpine;

		mPos = odeBodyGetPosition(bodyIDs[i]);

		float bone_size = boneSize[i] / 2;
		Vector3 ePos_i(mPos[0] + targetDirVector.x() * bone_size, mPos[1] + targetDirVector.y() * bone_size, mPos[2] + targetDirVector.z() * bone_size);

		return ePos_i;
	}

	Vector3 getBodyDirectionY(int i)
	{
		dMatrix3 R;
		ConstOdeQuaternion mQ;
		if (i == BodyName::BodyTrunkLower)
			mQ = odeBodyGetQuaternion(bodyIDs[BodyName::BodySpine]);
		else
			mQ = odeBodyGetQuaternion(bodyIDs[i]);
		dRfromQ(R, mQ);

		int targetDirection = 1; // alwayse in y direction
		int t_i = i;
		if (i == BodyName::BodyTrunkLower)
			t_i = BodyName::BodySpine;
		if (bodyTypes[t_i] == BodyType::BodyBox)
		{
			if (i == BodyName::BodyLeftArm || i == BodyName::BodyLeftShoulder || i == BodyName::BodyRightArm || i == BodyName::BodyRightShoulder)
				targetDirection = 0; // unless one of these bones is wanted
		}
		Vector3 targetDirVector(R[4 * 0 + targetDirection], R[4 * 1 + targetDirection], R[4 * 2 + targetDirection]);
		if (i == BodyName::BodyTrunkLower)
		{
			targetDirVector = -targetDirVector;
		}

		return targetDirVector;
	}

	float getClimberRadius()
	{
		return climberRadius;
	}

	float getClimberLegLegDis()
	{
		return climberLegLegDis;
	}

	float getClimberHandHandDis()
	{
		return climberHandHandDis;
	}

	int getJointBody(BodyName iBodyName)
	{
		return jointIDs[iBodyName - 1];
	}

	Vector3 computeCOM()
	{
		float totalMass=0;
		Vector3 result=Vector3::Zero();
		for (int i = 0; i < BodyNUM; i++)
		{
			float mass = odeBodyGetMass(bodyIDs[i]);
			Vector3 pos(odeBodyGetPosition(bodyIDs[i]));
			result+=mass*pos;
			totalMass+=mass;
		}
		return result/totalMass;
	}

	/////////////////////////////////////////// setting motor speed to control humanoid body /////////////

	void setMotorSpeed(int i, float iSpeed)
	{
		if (jointIDs[jointIDIndex[i]] == -1)
			return;
		if ((jointIDIndex[i] + 1) == BodyName::BodyHead || (jointIDIndex[i] + 1) == BodyName::BodyLeftHand || (jointIDIndex[i] + 1) == BodyName::BodyRightHand )
		{
			float angle=odeJointGetAMotorAngle(jointIDs[jointIDIndex[i]],jointAxisIndex[i]);
			iSpeed=-angle; //p control, keep head and wriat zero rotation
			//Vector3 dir_trunk = getBodyDirectionZ(BodyName::BodyTrunk);
			//Vector3 dir_head = getBodyDirectionZ(BodyName::BodyTrunk);
			//Vector3 diff_head_trunk = dir_trunk - dir_head;
			//diff_head_trunk[2] += FLT_EPSILON;
			//switch (jointAxisIndex[i])
			//{
			//case 0:// rotation about local x
			//	diff_head_trunk[0] = 0;
			//	diff_head_trunk.normalize();
			//	iSpeed = atan2(diff_head_trunk[1], diff_head_trunk[2]);
			//	break;
			//case 1:
			//	diff_head_trunk[1] = 0;
			//	diff_head_trunk.normalize();
			//	iSpeed = atan2(diff_head_trunk[0], diff_head_trunk[2]);
			//	break;
			//case 2:
			//	iSpeed = 0;
			//	break;
			//}
		}
		Vector2 mLimits = jointLimits[i];
		const float angleLimitBuffer=deg2rad*2.5f;
		if (jointTypes[jointIDIndex[i]] == JointType::ThreeDegreeJoint)
		{
			float cAngle = odeJointGetAMotorAngle(jointIDs[jointIDIndex[i]], jointAxisIndex[i]);
			float nAngle = cAngle + (float)(nPhysicsPerStep * iSpeed * timeStep);
			if (((nAngle < mLimits[0]+angleLimitBuffer) && iSpeed<0)
				|| ((nAngle > mLimits[1]+angleLimitBuffer) && iSpeed>0))
			{
				iSpeed = 0;
			}
			switch (jointAxisIndex[i])
			{
			case 0:
				odeJointSetAMotorParam(jointIDs[jointIDIndex[i]], dParamVel1, iSpeed);
				break;
			case 1:
				odeJointSetAMotorParam(jointIDs[jointIDIndex[i]], dParamVel2, iSpeed);
				break;
			case 2:
				odeJointSetAMotorParam(jointIDs[jointIDIndex[i]], dParamVel3, iSpeed);
				break;
			}
			return;
		}
		float cAngle = odeJointGetHingeAngle(jointIDs[jointIDIndex[i]]);
		float nAngle = cAngle + (float)(nPhysicsPerStep * iSpeed * timeStep);
		if (((nAngle < mLimits[0]+angleLimitBuffer) && iSpeed<0)
			|| ((nAngle > mLimits[1]+angleLimitBuffer) && iSpeed>0))
		{
			iSpeed = 0;
		}
		odeJointSetHingeParam(jointIDs[jointIDIndex[i]], dParamVel1, iSpeed);
		return;
	}
	
	void driveMotorToPose(int i, float targetAngle)
	{
		if (jointIDs[jointIDIndex[i]] == -1)
			return;
		if ((jointIDIndex[i] + 1) == BodyName::BodyHead || (jointIDIndex[i] + 1) == BodyName::BodyLeftHand || (jointIDIndex[i] + 1) == BodyName::BodyRightHand )
		{
			float angle=odeJointGetAMotorAngle(jointIDs[jointIDIndex[i]],jointAxisIndex[i]);
			odeJointSetAMotorParam(jointIDs[jointIDIndex[i]], dParamVel1, -angle);
			odeJointSetAMotorParam(jointIDs[jointIDIndex[i]], dParamVel2, -angle);
			odeJointSetAMotorParam(jointIDs[jointIDIndex[i]], dParamVel3, -angle);
		}
		Vector2 mLimits = jointLimits[i];
		if (jointTypes[jointIDIndex[i]] == JointType::ThreeDegreeJoint)
		{
			float cAngle = odeJointGetAMotorAngle(jointIDs[jointIDIndex[i]], jointAxisIndex[i]);
			odeJointSetAMotorParam(jointIDs[jointIDIndex[i]], dParamVel1+jointAxisIndex[i]*dParamGroup, (targetAngle-cAngle)/motorPoseInterpolationTime);
			return;
		}
		float cAngle = odeJointGetHingeAngle(jointIDs[jointIDIndex[i]]);
		odeJointSetHingeParam(jointIDs[jointIDIndex[i]], dParamVel1, (targetAngle-cAngle)/motorPoseInterpolationTime);
	}
	
	void setFmax(int joint, float fmax)
	{
		if (odeJointGetType(joint)==dJointTypeAMotor)
		{
			odeJointSetAMotorParam(joint,dParamFMax1,fmax);
			odeJointSetAMotorParam(joint,dParamFMax2,fmax);
			odeJointSetAMotorParam(joint,dParamFMax3,fmax);
		}
		else
		{
			odeJointSetHingeParam(joint,dParamFMax1,fmax);
		}
	}
	
	void setMotorGroupFmaxes(const float *fmax)
	{
		setFmax(getJointBody(BodySpine),std::max(torsoMinFMax,fmax[fmTorso]));
		//setFmax(getJointBody(BodyHead),fmax[fmTorso]);

		setFmax(getJointBody(BodyLeftShoulder),fmax[fmLeftArm]);
		setFmax(getJointBody(BodyLeftArm),fmax[fmLeftArm]);
		setFmax(getJointBody(BodyLeftHand),fmax[fmLeftArm]);

		setFmax(getJointBody(BodyLeftThigh),fmax[fmLeftLeg]);
		setFmax(getJointBody(BodyLeftLeg),fmax[fmLeftLeg]);
		setFmax(getJointBody(BodyLeftFoot),fmax[fmLeftLeg]);

		setFmax(getJointBody(BodyRightShoulder),fmax[fmRightArm]);
		setFmax(getJointBody(BodyRightArm),fmax[fmRightArm]);
		setFmax(getJointBody(BodyRightHand),fmax[fmRightArm]);

		setFmax(getJointBody(BodyRightThigh),fmax[fmRightLeg]);
		setFmax(getJointBody(BodyRightLeg),fmax[fmRightLeg]);
		setFmax(getJointBody(BodyRightFoot),fmax[fmRightLeg]);
	}

	inline static float odeVectorSquaredNorm(ConstOdeVector v)
	{
		return squared(v[0])+squared(v[1])+squared(v[2]);
	}
	
	float getMotorAppliedSqTorque(int i)
	{
		if (jointIDs[jointIDIndex[i]] == -1)
			return 0;
		return odeVectorSquaredNorm(odeJointGetAccumulatedTorque(jointIDs[jointIDIndex[i]],0))
			+ odeVectorSquaredNorm(odeJointGetAccumulatedTorque(jointIDs[jointIDIndex[i]],1));
	}
	
	Vector3 getForceVectorOnEndBody(ContactPoints iEndPos, int targetContext)
	{
		int jointIndex = iEndPos - ContactPoints::LeftLeg;
		if (getHoldBodyIDs(jointIndex, targetContext) >= 0)
		{
			/*int bodyId;
			switch (iEndPos)
			{
			case ContactPoints::LeftLeg:
				bodyId = bodyIDs[BodyName::BodyLeftFoot];
				break;
			case ContactPoints::RightLeg:
				bodyId = bodyIDs[BodyName::BodyRightFoot];
				break;
			case ContactPoints::LeftArm:
				bodyId = bodyIDs[BodyName::BodyLeftHand];
				break;
			case ContactPoints::RightArm:
				bodyId = bodyIDs[BodyName::BodyRightHand];
				break;
			default:
				break;
			}
			int jointType = dJointTypeBall;*/

			ConstOdeVector res = odeJointGetAccumulatedForce(jointHoldBallIDs[jointIndex], 0);
			Vector3 mOut(res[0],res[1],res[2]);
			/*if (mOut.norm() == 0)
			{
				res = odeJointGetAccumulatedForce(jointHoldBallIDs[jointIndex], 1);
				mOut[0] = res[0];
				mOut[1] = res[1];
				mOut[2] = res[2];
			}*/
			//if (iEndPos == ContactPoints::LeftLeg || iEndPos == ContactPoints::RightLeg)
			return -mOut;
			//else
			//	return -mOut;
		}
		return Vector3(0.0f,0.0f,0.0f);
	}
	
	Vector3 getForceVectorOnEndBody(int _body_index, int targetContext)
	{
		if (_body_index-1 >= 0 && _body_index-1 <= (int)jointIDs.size() - 1)
		{
			int jointIndex = jointIDs[_body_index-1];
			ConstOdeVector res = odeJointGetAccumulatedForce(jointIndex, 0);
			Vector3 mOut(res[0],res[1],res[2]);

			return -mOut;
		}
		return Vector3(0,0,0);
	}

	float getSqForceOnFingers(int targetContext)
	{
		/*float result=0;
		dVector3 f;
		odeBodyGetAccumulatedForce(bodyIDs[BodyName::BodyLeftHand],-1,f);
		result+=odeVectorSquaredNorm(f);
		odeBodyGetAccumulatedForce(bodyIDs[BodyName::BodyRightHand],-1,f);
		result+=odeVectorSquaredNorm(f);
		return result;*/
		
		float result=0;
		for (int j=2; j<4; j++)
		{
			if (getHoldBodyIDs(j, targetContext) >= 0)
			{
				int joint = jointHoldBallIDs[j];
				if (odeJointGetBody(joint,0) >= 0)
					result += odeVectorSquaredNorm(odeJointGetAccumulatedForce(joint,0));
				if (odeJointGetBody(joint,1) >= 0)
					result += odeVectorSquaredNorm(odeJointGetAccumulatedForce(joint,1));
			}
		}
		return result;
	}
	
	float getDesMotorAngleFromID(int i)
	{
		int bodyID = jointIDIndex[i] + 1;
		int axisID = jointAxisIndex[i];
		return getDesMotorAngle(bodyID, axisID);
	}

	float getDesMotorAngle(int &iBodyName, int &iAxisNum)
	{
		int jointIndex = iBodyName - 1;
		if (jointIndex > (int)(jointIDs.size()-1))
		{
			jointIndex = jointIDs.size()-1;
		}
		if (jointIndex < 0)
		{
			jointIndex = 0;
		}
		iBodyName = jointIndex + 1;

		if (jointTypes[jointIndex] == JointType::ThreeDegreeJoint)
		{
			if (iAxisNum > 2)
			{
				iAxisNum = 0;
			}
			if (iAxisNum < 0)
			{
				iAxisNum = 2;
			}
		}
		else
		{
			if (iAxisNum != 0)
			{
				iAxisNum = 0;
			}
		}

		int m_angle_index = 0;
		for (int b = 0; b < BodyNUM; b++)
		{
			for (int j = 0; j < jointTypes[b]; j++)
			{
				if (b == iBodyName - 1 && j == iAxisNum)
				{
					return desiredAnglesBones[m_angle_index];
				}
				m_angle_index++;
			}
		}
		return 0.0f;
	}

	void setMotorAngle(int &iBodyName, int &iAxisNum, float& dAngle)
	{
		int jointIndex = iBodyName - 1;
		if (jointIndex > (int)(jointIDs.size()-1))
		{
			jointIndex = jointIDs.size()-1;
		}
		if (jointIndex < 0)
		{
			jointIndex = 0;
		}
		iBodyName = jointIndex + 1;

		if (jointTypes[jointIndex] == JointType::ThreeDegreeJoint)
		{
			if (iAxisNum > 2)
			{
				iAxisNum = 0;
			}
			if (iAxisNum < 0)
			{
				iAxisNum = 2;
			}
		}
		else
		{
			if (iAxisNum != 0)
			{
				iAxisNum = 0;
			}
		}

		int m_angle_index = 0;
		for (int b = 0; b < BodyNUM; b++)
		{
			int mJointIndex = b - 1;
			if (mJointIndex < 0)
			{
				continue;
			}
			if (jointIDs[mJointIndex] == -1)
			{
				m_angle_index += jointTypes[mJointIndex];
				continue;
			}
			float source_angle = 0.0f;
			for (int axis = 0; axis < jointTypes[mJointIndex]; axis++)
			{
				if (jointTypes[mJointIndex] == JointType::ThreeDegreeJoint)
				{
					source_angle = odeJointGetAMotorAngle(jointIDs[mJointIndex],axis);
				}
				else
				{
					source_angle = odeJointGetHingeAngle(jointIDs[mJointIndex]);
				}

				if (b == iBodyName && axis == iAxisNum)
				{
					desiredAnglesBones[m_angle_index] = dAngle;
				}

				float iSpeed = desiredAnglesBones[m_angle_index] - source_angle;
				m_angle_index++;

				if (fabs(iSpeed) > maxSpeed)
				{
					iSpeed = fsign(iSpeed) * maxSpeed;
				}

				if (jointTypes[mJointIndex] == JointType::ThreeDegreeJoint)
				{
					switch (axis)
					{
					case 0:
						odeJointSetAMotorParam(jointIDs[mJointIndex], dParamVel1, iSpeed);
						break;
					case 1:
						odeJointSetAMotorParam(jointIDs[mJointIndex], dParamVel2, iSpeed);
						break;
					case 2:
						odeJointSetAMotorParam(jointIDs[mJointIndex], dParamVel3, iSpeed);
						break;
					}
				}
				else
				{
					odeJointSetHingeParam(jointIDs[mJointIndex], dParamVel1, iSpeed);
				}
			}
		}

		return;
	}

	void saveContextIn(BipedState& c)
	{
		for (int i = 0; i < BodyNUM; i++)
		{
			if (c.bodyStates.size() < BodyNUM)
			{
				c.bodyStates.push_back(BodyState());
			}
		}

		for (int i = 0; i < BodyNUM; i++)
		{
			c.bodyStates[i].setPos(getBonePosition(i));
			c.bodyStates[i].setAngle(getBoneAngle(i));
			c.bodyStates[i].setVel(getBoneLinearVelocity(i));
			c.bodyStates[i].setAVel(getBoneAngularVelocity(i));
			c.bodyStates[i].setBoneSize(boneSize[i]);
			c.bodyStates[i].setBodyType(bodyTypes[i]);
		}

		/*for (int i = 0; i < getJointSize(); i++)
		{
			if ((int)c.toDesAngles.size() < getJointSize())
			{
				c.toDesAngles.push_back(getJointAngle(i));
			}
			else
			{
				c.toDesAngles[i] = getJointAngle(i);
			}
		}*/

		return;
	}

	//////////////////////////////////////////// for drawing bodies ///////////////////////////////////////
	static void drawLine(Vector3& mP1, Vector3& mP2) // color should be set beforehand!
	{
		float p1[] = {mP1.x(),mP1.y(),mP1.z()};
		float p2[] = {mP2.x(),mP2.y(),mP2.z()};
		rcDrawLine(p1, p2);
		return;
	}

	static void drawCross(const Vector3& p)
	{
		float cross_size = 0.3f;
		float p1[] = {p.x() - cross_size / 2, p.y(), p.z()};
		float p2[] = {p.x() + cross_size / 2, p.y(), p.z()};
		rcDrawLine(p1, p2);

		p1[0] = p.x();
		p1[1] = p.y() - cross_size / 2;
		p2[0] = p.x();
		p2[1] = p.y() + cross_size / 2;
		rcDrawLine(p1, p2);

		p1[1] = p.y();
		p1[2] = p.z() - cross_size / 2;
		p2[1] = p.y();
		p2[2] = p.z() + cross_size / 2;
		rcDrawLine(p1, p2);

		return;
	}

	static void drawCube(Vector3& mCenter, float mCubeSize)
	{
		float p1[] = {mCenter.x() - mCubeSize, mCenter.y(), mCenter.z() - mCubeSize};
		float p2[] = {mCenter.x() - mCubeSize, mCenter.y(), mCenter.z() + mCubeSize};
		float p3[] = {mCenter.x() + mCubeSize, mCenter.y(), mCenter.z() + mCubeSize};
		float p4[] = {mCenter.x() + mCubeSize, mCenter.y(), mCenter.z() - mCubeSize};
		rcDrawLine(p1, p2);
		rcDrawLine(p2, p3);
		rcDrawLine(p3, p4);
		rcDrawLine(p4, p1);

		return;
	}

	void setColorConnectedBody(int bodyID, int targetContext)
	{
		Vector3 tColor(0,1,1);
		float transparency = 0.5f;
		switch (bodyID)
		{
		case BodyName::BodyLeftFoot:
			if (getHoldBodyIDs(0, targetContext) != -1)
				rcSetColor(tColor.x(),tColor.y(),tColor.z(),transparency);
			break;
		case BodyName::BodyRightFoot:
			if (getHoldBodyIDs(1, targetContext) != -1)
				rcSetColor(tColor.x(),tColor.y(),tColor.z(),transparency);
			break;
		case BodyName::BodyLeftHand:
			if (getHoldBodyIDs(2, targetContext) != -1)
				rcSetColor(tColor.x(),tColor.y(),tColor.z(),transparency);
			break;
		case BodyName::BodyRightHand:
			if (getHoldBodyIDs(3, targetContext) != -1)
				rcSetColor(tColor.x(),tColor.y(),tColor.z(),transparency);
			break;
		default:
			break;
		}
		return;
	}

	void mDrawStuff(int iControlledBody, int iControlledHold, int targetContext, bool whileOptimizing, bool debug_show)
	{
		if (!whileOptimizing)
			setCurrentOdeContext(masterContext);

		//dsSetTexture (DS_WOOD);

		for (int i = 0; i < BodyNUM; i++)
		{
			rcSetColor(1,1,1,1.0f);

			setColorConnectedBody(i, targetContext);

			if (((mGeomID[i] == iControlledBody && TestID == mEnumTestCaseClimber::TestCntroller) 
				|| (i == iControlledBody && TestID == mEnumTestCaseClimber::TestAngle)) && iControlledBody >= 0 && isTestClimber)
			{
				rcSetColor(0,1,0,1.0f);
			}
			if (bodyTypes[i] == BodyType::BodyBox)
			{
				float lx, ly, lz;
				odeGeomBoxGetLengths(mGeomID[i], lx, ly, lz);
				float sides[] = {lx, ly, lz};
				rcDrawBox(odeBodyGetPosition(bodyIDs[i]), odeBodyGetRotation(bodyIDs[i]), sides);
			}
			else if (bodyTypes[i] == BodyType::BodyCapsule)
			{
				float radius,length;
				odeGeomCapsuleGetParams(mGeomID[i], radius, length);
				rcDrawCapsule(odeBodyGetPosition(bodyIDs[i]), odeBodyGetRotation(bodyIDs[i]), length, radius);
			}
			else
			{
				float radius = odeGeomSphereGetRadius(mGeomID[i]);
				//rcDrawMarkerCircle(odeBodyGetPosition(bodyIDs[i]), radius);
				rcDrawSphere(odeBodyGetPosition(bodyIDs[i]), odeBodyGetRotation(bodyIDs[i]), radius);
			}
		}
		
		// draw wall and holds
		if (!whileOptimizing)
		{
			//dsSetTexture(DS_TEXTURE_NUMBER::DS_NONE);
			for (unsigned int i = 0; i < mENVGeomTypes.size(); i++)
			{
				if (mENVGeomTypes[i] == BodyType::BodyBox)
				{
					rcSetColor(1,1,1,0.5f);
					float lx, ly, lz;
					odeGeomBoxGetLengths(mENVGeoms[i], lx, ly, lz);
					float sides[] = {lx, ly, lz};
					rcDrawBox(odeGeomGetPosition(mENVGeoms[i]), odeGeomGetRotation(mENVGeoms[i]), sides); 
				}
				else
				{
					float radius = odeGeomSphereGetRadius(mENVGeoms[i]) / 2.0f;
					ConstOdeVector p = odeGeomGetPosition(mENVGeoms[i]);
					rcSetColor(1,1,0,0.25f);
					if (mENVGeoms[i] == iControlledHold && iControlledHold >= 0 && isTestClimber)
					{
						rcSetColor(0,1,0,0.5f);
						rcDrawSphere(p, odeGeomGetRotation(mENVGeoms[i]), radius);
					}
					
					float vPos[] = {p[0],p[1],p[2]};
					vPos[1] -= 0.015f;
					rcDrawMarkerCircle(vPos, 0.125f);

					if (debug_show)
					{
						Vector3 p1 = holds_body[i - startingHoldGeomsIndex].holdPos;
						p1[1] = -(holds_body[i - startingHoldGeomsIndex].size / 2 + holds_body[i - startingHoldGeomsIndex].size / 4);
						Vector3 p2 = p1 + (holds_body[i - startingHoldGeomsIndex].d_ideal) * 0.5f;
						rcSetColor(0,1,1);
						drawLine(p1, p2);
					}
				}
			}

			rcSetColor(0,1,0,1.0f);
			for (unsigned int i = 0; i < BodyNUM; i++)
			{
				Vector3 dir_i = getBodyDirectionY(i);

				Vector3 pos_i = getBonePosition(i);

				Vector3 n_pos_i = pos_i + 0.5f * dir_i;
				if (i == BodyName::BodyTrunk)
				{
					drawLine(pos_i, n_pos_i);
				}

				/*if (i == BodyName::BodyLeftFoot)
				{
					drawLine(pos_i, n_pos_i);
				}

				if (i == BodyName::BodyRightFoot)
				{
					drawLine(pos_i, n_pos_i);
				}*/
			}
		}

		return;
	}
	
	//////////////////////////////////////////// goal info ///////////////////////////////////////////////

	Vector3 getGoalPos()
	{
		return goal_pos;
	}

	//////////////////////////////////////////// create environment //////////////////////////////////////
	
	void createEnvironment(mDemoTestClimber DemoID, float wWidth, float wHeight, std::vector<std::vector<float>>& mHoldInfo)
	{
		Vector3 middleWallDegree = createWall(DemoID, wWidth, wHeight);

		startingHoldGeomsIndex = mENVGeoms.size();

		_wWidth = wWidth;
		_wHeight = wHeight;

		Vector3 cPos;
		if (mHoldInfo.size() > 0)
		{
			for (unsigned int i = 0; i < mHoldInfo.size(); i++)
			{
				switch (mHoldInfo[i].size())
				{
				case 2:
				case 7:
					cPos[0] = -wWidth / 2 + mHoldInfo[i][0];
					cPos[1] = 0.0f;
					cPos[2] = mHoldInfo[i][1];
					if (mHoldInfo[i].size() > 2)
						addHoldBodyToWorld(cPos, 
										   mHoldInfo[i][2],
										   Vector3(mHoldInfo[i][3], mHoldInfo[i][4], mHoldInfo[i][5]), 
										   mHoldInfo[i][6], 
										   0.25f);//, 
										   //Vector3(mHoldInfo[i][0], 0, mHoldInfo[i][1]));
					else
						addHoldBodyToWorld(cPos);
				break;
				case 9:
					cPos[0] = -wWidth / 2 + mHoldInfo[i][0];
					cPos[1] = -(mHoldInfo[i][8] / 4) + mHoldInfo[i][1];
					cPos[2] = mHoldInfo[i][2];
					
					addHoldBodyToWorld(cPos,
									   mHoldInfo[i][3],
									   Vector3(mHoldInfo[i][4], mHoldInfo[i][5], mHoldInfo[i][6]),
									   mHoldInfo[i][7],
									   mHoldInfo[i][8]);//,
									   //Vector3(mHoldInfo[i][0], mHoldInfo[i][1], mHoldInfo[i][2]));
					break;
				case 10:
					cPos[0] = -wWidth / 2 + mHoldInfo[i][0];
					cPos[1] = -(mHoldInfo[i][8] / 4) + mHoldInfo[i][1];
					cPos[2] = mHoldInfo[i][2];
					
					addHoldBodyToWorld(cPos,
									   mHoldInfo[i][3],
									   Vector3(mHoldInfo[i][4], mHoldInfo[i][5], mHoldInfo[i][6]),
									   mHoldInfo[i][7],
									   mHoldInfo[i][8],
									   //Vector3(mHoldInfo[i][0], mHoldInfo[i][1], mHoldInfo[i][2]),
									   mHoldInfo[i][9]);
					break;
				}
			}
			goal_pos = cPos;
			return;
		}

		float betweenHolds = 0.2f;
				
		Vector3 rLegPos;
		Vector3 lLegPos;

		float cHeightZ = 0.35f;
		float cWidthX = -FLT_MAX;

		float wall_1_z = 0.195f;
		float wall_1_x = 0.245f;
		float btwHolds1 = 0.2f;

		float wall_2_z = 1.37f;
		float wall_2_x = 0.25f;
		float btwHolds2_middle = 0.2f;
		float btwHolds2_topRow = 0.22f;

		float wall_3_z = 2.495f;
		float wall_3_x = 0.3f;
		float btwHolds3 = 0.17f;

		switch (DemoID)
		{
		case mDemoTestClimber::DemoRoute1:
			cPos = getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
			cPos[1] = -0.045f;
			cPos[2] = wall_1_z;
			addHoldBodyToWorld(cPos); // leftLeg
			lLegPos = cPos;

			rLegPos = cPos;
			rLegPos[0] += 4 * btwHolds1;
			addHoldBodyToWorld(rLegPos); // rightLeg

			cPos[2] = wall_2_z;
			addHoldBodyToWorld(cPos); // LeftHand

			cPos[0] += 3 * btwHolds2_middle;
			addHoldBodyToWorld(cPos); // RightHand

			cPos = lLegPos;
			cPos[2] += 3 * btwHolds1;
			cPos[0] += 2 * btwHolds1;
			addHoldBodyToWorld(cPos); // helper 1

			cPos = lLegPos;
			cPos[2] = wall_2_z + 3 * btwHolds2_middle + btwHolds2_topRow;
			cPos[0] += btwHolds2_topRow;
			addHoldBodyToWorld(cPos); // helper 2

			cPos = lLegPos;
			cPos[2] = wall_3_z + btwHolds3;
			cPos[0] += (4 * btwHolds3 - 2 * btwHolds1);
			addHoldBodyToWorld(cPos); // helper 3

			cPos[2] += 3 * btwHolds3;
			cPos[0] += (1 * btwHolds3);
			addHoldBodyToWorld(cPos); // goal

			break;
		case mDemoTestClimber::DemoRoute2:
			cPos = getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
			cPos[1] = 0.0f;
			cPos[2] = wall_1_z;
			addHoldBodyToWorld(cPos); // leftLeg
			lLegPos = cPos;

			rLegPos = cPos;
			rLegPos[0] += 2 * btwHolds1;
			addHoldBodyToWorld(rLegPos); // rightLeg

			cPos[2] = wall_2_z;
			addHoldBodyToWorld(cPos); // LeftHand

			cPos[0] += 2 * btwHolds2_middle;
			addHoldBodyToWorld(cPos); // RightHand

			cPos = lLegPos;
			cPos[2] += 3 * btwHolds1;
			cPos[0] += 3 * btwHolds1;
			addHoldBodyToWorld(cPos); // helper 1

			cPos = lLegPos;
			cPos[2] = wall_2_z;
			cPos[0] += 9 * btwHolds2_middle;
			addHoldBodyToWorld(cPos); // helper 2

			cPos = lLegPos;
			cPos[2] = wall_2_z + 3 * btwHolds2_middle;
			cPos[0] += (5 * btwHolds2_middle);
			addHoldBodyToWorld(cPos); // helper 3

			cPos = lLegPos;
			cPos[2] = wall_2_z + 3 * btwHolds2_middle + btwHolds2_topRow;
			cPos[0] += (2 * btwHolds2_middle + btwHolds2_topRow);
			addHoldBodyToWorld(cPos); // helper 4

			cPos = lLegPos;
			cPos[2] = wall_3_z + 2 * btwHolds3;
			cPos[0] += (9 * btwHolds3);
			addHoldBodyToWorld(cPos); // helper 5

			cPos[2] += 2 * btwHolds3;
			addHoldBodyToWorld(cPos); // goal

			break;
		case mDemoTestClimber::DemoRoute3:
			cPos = getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
			cPos[1] = 0.0f;
			cPos[2] = wall_1_z;
			addHoldBodyToWorld(cPos); // leftLeg
			lLegPos = cPos;

			rLegPos = cPos;
			rLegPos[0] += 2 * btwHolds1;
			addHoldBodyToWorld(rLegPos); // rightLeg

			cPos[2] = wall_2_z;
			addHoldBodyToWorld(cPos); // LeftHand

			cPos[0] += 2 * btwHolds2_middle;
			addHoldBodyToWorld(cPos); // RightHand

			cPos = lLegPos;
			cPos[2] += 3 * btwHolds1;
			cPos[0] += 3 * btwHolds1;
			addHoldBodyToWorld(cPos); // helper 1

			cPos = lLegPos;
			cPos[2] = wall_2_z + 3 * btwHolds2_middle;
			cPos[0] += (5 * btwHolds2_middle);
			addHoldBodyToWorld(cPos); // helper 3

			cPos = lLegPos;
			cPos[2] = wall_2_z + 3 * btwHolds2_middle + btwHolds2_topRow;
			cPos[0] += (2 * btwHolds2_middle + btwHolds2_topRow);
			addHoldBodyToWorld(cPos); // helper 4

			cPos = lLegPos;
			cPos[2] = wall_3_z + 2 * btwHolds3;
			cPos[0] += (9 * btwHolds3);
			addHoldBodyToWorld(cPos); // helper 5

			cPos[2] += 2 * btwHolds3;
			addHoldBodyToWorld(cPos); // goal

			break;
		case mDemoTestClimber::DemoLongWall:
			rLegPos = getEndPointPosBones(SimulationContext::BodyName::BodyRightLeg);
			lLegPos = getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
			
			while (cHeightZ < 10.0f)
			{
				float cDisX = 0.0f;
				float _minZ = FLT_MAX;
				while (cDisX < 1.0f)
				{
					float r1 = (float)rand() / (float)RAND_MAX;
					float r2 = (float)rand() / (float)RAND_MAX;
					cPos = Vector3(lLegPos.x() - 0.4f + r1 * 0.2f + cDisX, 0.0f, cHeightZ + 0.3f * r2);
					addHoldBodyToWorld(cPos);
					cDisX += 0.8f;
					_minZ = min<float>(_minZ, cHeightZ + 0.3f * r2);
					cWidthX = max<float>(cWidthX, cPos.x());
				}
				cHeightZ = _minZ + climberRadius / 2.5f;
			}
			cHeightZ = 0.35f;
			while (cHeightZ < 10.0f)
			{
				float cDisX = 0.0f;
				float _minZ = FLT_MAX;
				while (cDisX < 1.0f)
				{
					float r1 = (float)rand() / (float)RAND_MAX;
					float r2 = (float)rand() / (float)RAND_MAX;
					cPos = Vector3(cWidthX + climberLegLegDis - 0.4f + r1 * 0.2f + cDisX, 0.0f, cHeightZ + 0.3f * r2);
					addHoldBodyToWorld(cPos);
					cDisX += 0.8f;
					_minZ = min<float>(_minZ, cHeightZ + 0.3f * r2);
				}
				cHeightZ = _minZ + climberRadius / 2.5f;
			}

			break;
		case mDemoTestClimber::Demo45Wall:
			cPos = getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
			cPos[1] = 0.0f;
			cPos[2] = 0.35f;
			addHoldBodyToWorld(cPos); // leftLeg

			rLegPos = cPos;
			rLegPos[0] += 2 * betweenHolds;
			addHoldBodyToWorld(rLegPos); // rightLeg

			cPos[2] += 3 * betweenHolds + 0.36f;
			addHoldBodyToWorld(cPos); // LeftHand

			cPos[0] += 2 * betweenHolds;
			addHoldBodyToWorld(cPos); // RightHand

			cPos[2] = middleWallDegree.y() + betweenHolds * sinf(middleWallDegree.z());
			cPos[1] += -betweenHolds * cosf(middleWallDegree.z());
			addHoldBodyToWorld(cPos); // RightHand 1

			cPos[0] -= 2 * betweenHolds;
			addHoldBodyToWorld(cPos); // LeftHand 1

			cPos[2] += 4 * betweenHolds * sinf(middleWallDegree.z());
			cPos[1] += -4 * betweenHolds * cosf(middleWallDegree.z());
			addHoldBodyToWorld(cPos); // LeftHand 2

			cPos[0] += 2 * betweenHolds;
			addHoldBodyToWorld(cPos); // RightHand 2

			cPos[2] += 4 * betweenHolds * sinf(middleWallDegree.z());
			cPos[1] += -4 * betweenHolds * cosf(middleWallDegree.z());
			addHoldBodyToWorld(cPos); // LeftHand 2

			cPos[0] -= 2 * betweenHolds;
			addHoldBodyToWorld(cPos); // RightHand 2

			cPos[2] += 4 * betweenHolds * sinf(middleWallDegree.z());
			cPos[1] += -4 * betweenHolds * cosf(middleWallDegree.z());
			addHoldBodyToWorld(cPos); // LeftHand 2

			cPos[0] += 2 * betweenHolds;
			addHoldBodyToWorld(cPos); // RightHand 2
			break;
		default:
			break;
		};

		goal_pos = cPos;
	}

	Vector3 crossProduct(Vector3 a, Vector3 b)
	{
		a.normalize();
		b.normalize();
		Vector3 _n(a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]);
		_n.normalize();
		return _n;
	}

	Vector3 getGeomAxisDirection(int geomID, int axis)
	{
		dMatrix3 R;
		dReal mQ[4];
		odeGeomGetQuaternion(geomID, mQ);
		
		dRfromQ(R, mQ);

		int targetDirection = axis; // alwayse in y direction		
		return Vector3(R[4 * 0 + targetDirection], R[4 * 1 + targetDirection], R[4 * 2 + targetDirection]);
	}

	Vector3 getAxisDirectionFrom(const dMatrix3 R, int axis)
	{
		int targetDirection = axis; // alwayse in y direction		
		return Vector3(R[4 * 0 + targetDirection], R[4 * 1 + targetDirection], R[4 * 2 + targetDirection]);
	}

	void createRotateSurface(Vector3 v1, Vector3 v2, Vector3 v3, Vector3 lookAtPos)
	{
		//float wallTickness = 2 * boneRadius;

		//Vector3 _pos = (0.5f * (v1 + v2) + v3) / 2.0f;

		//Vector3 oN(0,1,0);
		//Vector3 nN = crossProduct(v2 - v1, v3 - v1);
		//Vector3 dirSurface = (lookAtPos-_pos).normalized();

		//if (getAbsAngleBtwVectors(dirSurface, nN) > PI / 2)
		//{
		//	nN = -nN;
		//}

		//float _width = (v2 - v1).norm();
		//float _height = (0.5f * (v1+v2) - v3).norm();

		//int cGeomID = odeCreateBox(_width, wallTickness, _height);

		//_pos = _pos - (0.5f * wallTickness) * nN;
		//odeGeomSetPosition(cGeomID, _pos[0], _pos[1], _pos[2]);

		//Vector3 rV = crossProduct(oN, nN);
		//float angle = getAbsAngleBtwVectors(oN, nN);

		//dMatrix3 R1;
		//dRFromAxisAndAngle(R1, rV[0], rV[1], rV[2], angle);

		//Vector3 xDir = getAxisDirectionFrom(R1, 0);
		//Vector3 yDir = getAxisDirectionFrom(R1, 1);
		//Vector3 zDir = getAxisDirectionFrom(R1, 2);

		//dMatrix3 R2;
		//Vector3 nZDir = (v3 - (v2 + v1) / 2.0f).normalized();
		//angle = getAbsAngleBtwVectors(zDir, nZDir);
		//
		//nN = yDir;
		//dRFromAxisAndAngle(R2, nN[0], nN[1], nN[2], angle);

		//dMatrix3 fR;
		//dMultiply0(fR, R2, R1, 3, 3, 3);
		//fR[3] = 0.0f;
		//fR[7] = 0.0f;
		//fR[11] = 0.0f;

		//xDir = getAxisDirectionFrom(fR, 0);
		//yDir = getAxisDirectionFrom(fR, 1);
		//zDir = getAxisDirectionFrom(fR, 2);

		//if (!((zDir - nZDir).norm() < 0.01f && (yDir - nN).norm() < 0.01f))
		//{
		//	dRFromAxisAndAngle(R2, nN[0], nN[1], nN[2], -angle);

		//	dMultiply0(fR, R2, R1, 3, 3, 3);
		//	fR[3] = 0.0f;
		//	fR[7] = 0.0f;
		//	fR[11] = 0.0f;

		//	xDir = getAxisDirectionFrom(fR, 0);
		//	yDir = getAxisDirectionFrom(fR, 1);
		//	zDir = getAxisDirectionFrom(fR, 2);
		//}

		//odeGeomSetRotation(cGeomID, fR);

		//xDir = getGeomAxisDirection(cGeomID, 0);
		//yDir = getGeomAxisDirection(cGeomID, 1);
		//zDir = getGeomAxisDirection(cGeomID, 2);

		//mENVGeoms.push_back(cGeomID);
		//mENVGeomTypes.push_back(BodyType::BodyBox);

		//odeGeomSetCategoryBits(cGeomID, unsigned long(0));
		//odeGeomSetCollideBits (cGeomID, unsigned long(0x7FFF)); //

		float wallTickness = 2 * boneRadius;

		Vector3 _pos = (0.5f * (v1 + v2) + v3) / 2.0f;

		Vector3 oN(0,1,0);
		Vector3 nN = crossProduct(v2 - v1, v3 - v1);
		Vector3 dirSurface = (lookAtPos-_pos).normalized();

		float _width = (v2 - v1).norm();
		float _height = (0.5f * (v1+v2) - v3).norm();

		int cGeomID = odeCreateBox(_width, wallTickness, _height);

		if (getAbsAngleBtwVectors(dirSurface, nN) > PI / 2)
		{
			_pos = _pos + (0.5f * wallTickness) * nN;
		}
		else
		{
			_pos = _pos - (0.5f * wallTickness) * nN;
		}
		odeGeomSetPosition(cGeomID, _pos[0], _pos[1], _pos[2]);

		Vector3 rV = crossProduct(oN, nN);
		float angle = getAbsAngleBtwVectors(oN, nN);

		dMatrix3 R1;
		dRFromAxisAndAngle(R1, rV[0], rV[1], rV[2], angle);

		Vector3 xDir = getAxisDirectionFrom(R1, 0);
		Vector3 yDir = getAxisDirectionFrom(R1, 1);
		Vector3 zDir = getAxisDirectionFrom(R1, 2);

		dMatrix3 R2;
		Vector3 nXDir = (v2 - v1).normalized();
		angle = getAbsAngleBtwVectors(xDir, nXDir);
		
		//nN = yDir;
		dRFromAxisAndAngle(R2, nN[0], nN[1], nN[2], angle);

		dMatrix3 fR;
		dMultiply0(fR, R2, R1, 3, 3, 3);
		fR[3] = 0.0f;
		fR[7] = 0.0f;
		fR[11] = 0.0f;

		xDir = getAxisDirectionFrom(fR, 0);
		yDir = getAxisDirectionFrom(fR, 1);
		zDir = getAxisDirectionFrom(fR, 2);

		if ((xDir - nXDir).norm() > 0.001f)
		{
			dRFromAxisAndAngle(R2, nN[0], nN[1], nN[2], -angle);

			dMultiply0(fR, R2, R1, 3, 3, 3);
			fR[3] = 0.0f;
			fR[7] = 0.0f;
			fR[11] = 0.0f;

			xDir = getAxisDirectionFrom(fR, 0);
			yDir = getAxisDirectionFrom(fR, 1);
			zDir = getAxisDirectionFrom(fR, 2);
		}

		odeGeomSetRotation(cGeomID, fR);

		xDir = getGeomAxisDirection(cGeomID, 0);
		yDir = getGeomAxisDirection(cGeomID, 1);
		zDir = getGeomAxisDirection(cGeomID, 2);

		mENVGeoms.push_back(cGeomID);
		mENVGeomTypes.push_back(BodyType::BodyBox);

		odeGeomSetCategoryBits(cGeomID, unsigned long(0));
		odeGeomSetCollideBits (cGeomID, unsigned long(0x7FFF)); //

		return;
	}

	void createWallAroundPyramid(Vector3 _from, Vector3 _to, Vector3 _boundary)
	{
		float dis = min((_boundary - _from).norm(), (_boundary - _to).norm());
		Vector3 _dir = (_to - _from).normalized();
		Vector3 dir(-_dir[2],_dir[1],_dir[0]);
		if (dis > 0 && dir.norm() > 0)
			createRotateSurface(_from, _to, 0.5f * (_from + _to) + dis * dir, 0.5f * (_from + _to) + Vector3(0,-1,0));
		return;
	}

	bool createPyramid(std::vector<Vector3>& _boundaries, std::vector<Vector3>& _points)
	{
		Vector3 pp1 = _points[0];
		Vector3 pp2 = _points[1];
		Vector3 pp3 = _points[2];
		Vector3 pp4 = _points[3];
		Vector3 _midPos = (pp1 + pp2 + pp3 + pp4) / 4.0f;
		Vector3 midPos = _points[4];
		
		createRotateSurface(pp1, pp2, midPos, _midPos);
		createRotateSurface(pp2, pp3, midPos, _midPos);
		createRotateSurface(pp3, pp4, midPos, _midPos);
		createRotateSurface(pp4, pp1, midPos, _midPos);

		Vector3 up = pp1; // search for max z
		Vector3 right = pp2; // search for max x
		Vector3 down = pp3; // search for min z
		Vector3 left = pp1; // search for min x

		for (int i = 0; i < 4; i++)
		{
			if (_points[i].z() > up.z())
			{
				up = _points[i];
			}
			if (_points[i].x() > right.x())
			{
				right = _points[i];
			}
			if (_points[i].z() < down.z())
			{
				down = _points[i];
			}
			if (_points[i].x() < left.x())
			{
				left = _points[i];
			}
		}

		Vector3 up_right(max(up[0],right[0]), 0.0f, max(up[2],right[2]));
		createWallAroundPyramid(up, right, up_right);

		Vector3 down_right(max(down[0],right[0]), 0.0f, min(down[2],right[2]));
		createWallAroundPyramid(right, down, down_right);
		
		Vector3 down_left(min(down[0],left[0]), 0.0f, min(down[2],left[2]));
		createWallAroundPyramid(down, left, down_left);

		Vector3 left_up(min(up[0],left[0]), 0.0f, max(up[2],left[2]));
		createWallAroundPyramid(left, up, left_up);

		_boundaries.push_back(up);
		_boundaries.push_back(right);
		_boundaries.push_back(down);
		_boundaries.push_back(left);

		return true;
	}

	void createWallFromTo(Vector3 _from, Vector3 _to)
	{
		float _width = _to[0] - _from[0];
		float _height = _to[2] - _from[2];
		int cGeomID = odeCreateBox(_width, 2 * boneRadius, _height);

		Vector3 _pos = (_to + _from) * 0.5f;

		odeGeomSetPosition(cGeomID, _pos[0], (1.5) * boneRadius + mBiasWallY + 0.5f * boneRadius, _pos[2]);
			
		mENVGeoms.push_back(cGeomID);
		mENVGeomTypes.push_back(BodyType::BodyBox);

		odeGeomSetCategoryBits(cGeomID, unsigned long(0));
		odeGeomSetCollideBits (cGeomID, unsigned long(0x7FFF)); //
	}

	Vector3 createWall(mDemoTestClimber DemoID, float wWidth, float wHeight)
	{
		if (DemoID != mDemoTestClimber::Demo45Wall)
		{
			mFileHandler mWholeInfo("ClimberInfo\\wholeInfo.txt");
			std::vector<std::vector<float>> xyz_points;
			mWholeInfo.readFile(xyz_points);
			std::vector<Vector3> _boundaries1;
			std::vector<Vector3> _boundaries2;

			std::vector<Vector3> _iPoints;
			for (int i = 0; i < 5; i++)
			{
				_iPoints.push_back(Vector3(-wWidth/2 + xyz_points[i][0], xyz_points[i][1], xyz_points[i][2]));
			}
			createPyramid(_boundaries1, _iPoints);
			_iPoints.clear();

			for (int i = 5; i < 10; i++)
			{
				_iPoints.push_back(Vector3(-wWidth/2 + xyz_points[i][0], xyz_points[i][1], xyz_points[i][2]));
			}
			createPyramid(_boundaries2, _iPoints);
			_iPoints.clear();

			wWidth = 1.75 * wWidth;
			wHeight = 1.2 * wHeight;

			std::vector<Vector3> maxBoundary;
			std::vector<Vector3> minBoundary;

			if (_boundaries1[2][2] > _boundaries2[0][2])
			{
				maxBoundary = _boundaries1;
				minBoundary = _boundaries2;
			}
			else
			{
				maxBoundary = _boundaries2;
				minBoundary = _boundaries1;
			}

			createWallFromTo(Vector3(-wWidth/2,0,0), Vector3(wWidth/2,0,minBoundary[2][2]));
			createWallFromTo(Vector3(minBoundary[1][0],0,minBoundary[2][2]), Vector3(wWidth/2,0,minBoundary[0][2]));
			createWallFromTo(Vector3(-wWidth/2,0,minBoundary[2][2]), Vector3(minBoundary[3][0],0,minBoundary[0][2]));

			createWallFromTo(Vector3(-wWidth/2,0,minBoundary[0][2]), Vector3(wWidth/2,0,maxBoundary[2][2]));
			createWallFromTo(Vector3(maxBoundary[1][0],0,maxBoundary[2][2]), Vector3(wWidth/2,0,maxBoundary[0][2]));
			createWallFromTo(Vector3(-wWidth/2,0,maxBoundary[2][2]), Vector3(maxBoundary[3][0],0,maxBoundary[0][2]));

			createWallFromTo(Vector3(-wWidth/2,0,maxBoundary[0][2]), Vector3(wWidth/2,0,wHeight));

			return Vector3(0.0f,0.0f,0.0f);
		}
		else
		{
			// starting from vertical wall
			int cGeomID = odeCreateBox(5, 2 * boneRadius, 2);
			odeGeomSetPosition(cGeomID, 0, (1.5) * boneRadius + mBiasWallY + 0.5f * boneRadius, 1.0f);
			mENVGeoms.push_back(cGeomID);
			mENVGeomTypes.push_back(BodyType::BodyBox);

			// going to 45 degree wall
			float l = 5.0f;
			float angle = PI/4;
			float x_b = ((1.5) * boneRadius + mBiasWallY + 0.5f * boneRadius); // -boneRadius -0.5f * l * sinf(angle)
			cGeomID = odeCreateBox(5, 2 * boneRadius, l);
			odeGeomSetPosition(cGeomID, 0,  x_b - 0.5f * (l) * sinf(angle), 2 + 0.5f * boneRadius + 0.5f * (l) * cosf(angle)); // - tanf(angle) * x_b
			
			dMatrix3 R;
			dRFromAxisAndAngle(R, 1, 0, 0, angle);
			odeGeomSetRotation(cGeomID, R);

			mENVGeoms.push_back(cGeomID);
			mENVGeomTypes.push_back(BodyType::BodyBox);

			return Vector3(0.0f, 2.0f, PI/2 - angle);
		}

		return Vector3(0.0f,0.0f,0.0f);
	}

	void addHoldBodyToWorld(Vector3 cPos, float f_ideal = 1000.0f, Vector3 d_ideal = Vector3(0, 0,-1), float k = 1.0f, float _s = holdSize / 2, int _HoldPushPullMode = 0) //Vector3 _notTransformedPos = Vector3(0, 0,0), int _HoldPushPullMode = 0)
	{
		float iX = cPos.x(), iY = cPos.y(), iZ = cPos.z();

		int cGeomID = odeCreateSphere(_s);

		odeGeomSetPosition(cGeomID, iX, iY, iZ);

		mENVGeoms.push_back(cGeomID);
		mENVGeomTypes.push_back(BodyType::BodySphere);

		odeGeomSetCollideBits (cGeomID, unsigned long(0x8000)); // do not collide with anything!, but used for rayCasting
		odeGeomSetCategoryBits (cGeomID, unsigned long(0x8000)); // do not collide with anything!, but used for rayCasting

		holds_body.push_back(holdContent(Vector3(iX, iY, iZ), f_ideal, d_ideal, k, _s, cGeomID, _HoldPushPullMode));//_notTransformedPos, _HoldPushPullMode));
	}

	int createJointType(float pX, float pY, float pZ, int pBodyNum, int cBodyNum, JointType iJ = JointType::BallJoint)
	{
		//float positionSpring = 10000.0f, stopSpring = 20000.0f, damper = 1.0f, maximumForce = 200.0f;
		float positionSpring = 5000.0f, stopSpring = 5000.0f, damper = 1.0f;
		
		float kp = positionSpring; 
        float kd = damper; 

        float erp = timeStep * kp / (timeStep * kp + kd);
        float cfm = 1.0f / (timeStep * kp + kd);

        float stopDamper = 1.0f; //stops best when critically damped
        float stopErp = timeStep * stopSpring / (timeStep * kp + stopDamper);
        float stopCfm = 1.0f / (timeStep * stopSpring + stopDamper);

		int jointID = -1;
		switch (iJ)
		{
			case JointType::BallJoint:
				jointID = odeJointCreateBall();
			break;
			case JointType::ThreeDegreeJoint:
				jointID = odeJointCreateAMotor();
			break;
			case JointType::TwoDegreeJoint:
				jointID = odeJointCreateHinge2();
			break;
			case JointType::OneDegreeJoint:
				jointID = odeJointCreateHinge();
			break;
			case JointType::FixedJoint:
				jointID = odeJointCreateFixed();
			break;
			default:
				break;
		}
		
		if (pBodyNum >= 0)
		{
			if (cBodyNum < BodyNUM && cBodyNum >= 0)
			{
				odeJointAttach(jointID, bodyIDs[pBodyNum], bodyIDs[cBodyNum]);
			}

			if (iJ == JointType::FixedJoint)
				odeJointSetFixed(jointID);
		}
		else
		{
			if (cBodyNum < BodyNUM && cBodyNum >= 0)
			{
				odeJointAttach(jointID, 0, bodyIDs[cBodyNum]);
			}

			if (iJ == JointType::FixedJoint)
				odeJointSetFixed(jointID);
		}

		float angle = 0;
		switch (iJ)
		{
			case JointType::BallJoint:
				odeJointSetBallAnchor(jointID, pX, pY, pZ);
			break;
			case JointType::ThreeDegreeJoint:
				odeJointSetAMotorNumAxes(jointID, 3);

				if (pBodyNum >= 0)
				{
					odeJointSetAMotorAxis(jointID, 0, 1, 0, 1, 0);
					odeJointSetAMotorAxis(jointID, 2, 2, 1, 0, 0);

					if (cBodyNum == BodyName::BodyLeftShoulder || cBodyNum == BodyName::BodyRightShoulder)
					{
						odeJointSetAMotorAxis(jointID, 0, 1, 0, 1, 0);
						odeJointSetAMotorAxis(jointID, 2, 2, 0, 0, 1);
					}
				}
				else
				{

					odeJointSetAMotorAxis(jointID, 0, 0, 1, 0, 0);
					odeJointSetAMotorAxis(jointID, 2, 2, 0, 0, 1);
				}

				odeJointSetAMotorMode(jointID, dAMotorEuler);

				odeJointSetAMotorParam(jointID, dParamFMax1, maximumForce);
				odeJointSetAMotorParam(jointID, dParamFMax2, maximumForce);
				odeJointSetAMotorParam(jointID, dParamFMax3, maximumForce);

				odeJointSetAMotorParam(jointID, dParamVel1, 0);
				odeJointSetAMotorParam(jointID, dParamVel2, 0);
				odeJointSetAMotorParam(jointID, dParamVel3, 0);

				odeJointSetAMotorParam(jointID, dParamCFM1, cfm);
				odeJointSetAMotorParam(jointID, dParamCFM2, cfm);
				odeJointSetAMotorParam(jointID, dParamCFM3, cfm);

				odeJointSetAMotorParam(jointID, dParamERP1, erp);
				odeJointSetAMotorParam(jointID, dParamERP2, erp);
				odeJointSetAMotorParam(jointID, dParamERP3, erp);

				odeJointSetAMotorParam(jointID, dParamStopCFM1, stopCfm);
				odeJointSetAMotorParam(jointID, dParamStopCFM2, stopCfm);
				odeJointSetAMotorParam(jointID, dParamStopCFM3, stopCfm);

				odeJointSetAMotorParam(jointID, dParamStopERP1, stopErp);
				odeJointSetAMotorParam(jointID, dParamStopERP2, stopErp);
				odeJointSetAMotorParam(jointID, dParamStopERP3, stopErp);

				odeJointSetAMotorParam(jointID, dParamFudgeFactor1, -1);
				odeJointSetAMotorParam(jointID, dParamFudgeFactor2, -1);
				odeJointSetAMotorParam(jointID, dParamFudgeFactor3, -1);
				
			break;
			case JointType::TwoDegreeJoint:
				odeJointSetHinge2Anchor(jointID, pX, pY, pZ);
				odeJointSetHinge2Axis(jointID, 1, 0, 1, 0);
				odeJointSetHinge2Axis(jointID, 2, 0, 0, 1);

				odeJointSetHinge2Param(jointID, dParamFMax1, maximumForce);
				odeJointSetHinge2Param(jointID, dParamFMax2, maximumForce);

				odeJointSetHinge2Param(jointID, dParamVel1, 0);
				odeJointSetHinge2Param(jointID, dParamVel2, 0);

				odeJointSetHinge2Param(jointID, dParamCFM1, cfm);
				odeJointSetHinge2Param(jointID, dParamCFM2, cfm);

				odeJointSetHinge2Param(jointID, dParamStopERP1, stopErp);
				odeJointSetHinge2Param(jointID, dParamStopERP2, stopErp);
				odeJointSetHinge2Param(jointID, dParamFudgeFactor1, -1);
				odeJointSetHinge2Param(jointID, dParamFudgeFactor2, -1);
			break;
			case JointType::OneDegreeJoint:
				odeJointSetHingeAnchor(jointID, pX, pY, pZ);
				if (cBodyNum == BodyName::BodyLeftLeg || cBodyNum == BodyName::BodyRightLeg || cBodyNum == BodyName::BodyRightFoot || cBodyNum == BodyName::BodyLeftFoot)
				{
					odeJointSetHingeAxis(jointID, 1, 0, 0);
				}
				else
				{
					odeJointSetHingeAxis(jointID, 0, 0, 1);
				}
				
				odeJointSetHingeParam(jointID, dParamFMax, maximumForce);

				odeJointSetHingeParam(jointID, dParamVel, 0);

				odeJointSetHingeParam(jointID, dParamCFM, cfm);

				odeJointSetHingeParam(jointID, dParamERP, erp);

				odeJointSetHingeParam(jointID, dParamStopCFM, stopCfm);

				odeJointSetHingeParam(jointID, dParamStopERP, stopErp);

				odeJointSetHingeParam(jointID, dParamFudgeFactor, -1);
			break;
			default:
				break;
		}

		return jointID;
	}

	Vector3 getVectorForm(std::vector<float>& _iV)
	{
		return Vector3(_iV[0], _iV[1], _iV[2]);
	}

	void createHumanoidBody(float pX, float pY, float dHeight, float dmass, std::vector<std::vector<float>>& _readFileClimberKinectInfo)
	{
		float current_height = 1.7758f; // with scale of one the created height is calculated in current_height
		float scale = dHeight / current_height;

		float ArmLegWidth = (0.75f * boneRadius) * scale;
		float feetWidth = ArmLegWidth;
		// trunk (root body without parent)
		float trunk_length = (0.1387f + 0.1363f) * scale; // done
		// spine
		float spine_length = (0.1625f + 0.091f) * scale; // done
		// thigh
		float thigh_length = (0.4173f) * scale; // done
		float dis_spine_leg_x = (0.09f) * scale;
		float dis_spine_leg_z = (0.00f) * scale;
		// leg
		float leg_length = (0.39f) * scale; // done
		// foot
		float feet_length = (0.08f + 0.127f + 0.05f) * scale; // done
		// shoulder
		float handShortenAmount = 0.025f;
		float dis_spine_shoulder_x = (0.06f + 0.123f) * scale;
		float dis_spine_shoulder_z = trunk_length - (0.1363f + 0.101f) * scale;
		float shoulder_length = (0.276f + handShortenAmount) * scale; // done
		// arm
		float arm_length = (0.278f) * scale; // done
		// hand
		float handsWidth = 0.9f * ArmLegWidth;
		float hand_length = (0.1f + 0.05f + 0.03f + 0.025f - handShortenAmount) * scale; // done
		// head
		float head_length = (0.21f + 0.04f) * scale; // done
		float dis_spine_head_z = (0.04f) * scale; // done
		float HeadWidth = (0.85f * boneRadius) * scale;

		Vector3 midPos;
		if (_readFileClimberKinectInfo.size() > 0)
		{
			enum mReadFileBoneType{_head = 0, _neck = 1, _spineshoulder = 2,
								   _spinemid = 3, _spinebase = 4, _shoulderright = 5,
								   _shoulderleft = 6, _hipright = 7, _hipleft = 8,
								   _elbowright = 9, _elbowleft = 10, _wristright = 11,
								   _wristleft = 12, _handtipright = 13, _handtipleft = 14,
								   _kneeright = 15, _kneeleft = 16, _ankleright = 17,
								   _ankleleft = 18, _feetright = 19, _feetleft = 20,
								   _groundfeetright = 21, _groundfeetleft = 22};
			trunk_length = (getVectorForm(_readFileClimberKinectInfo[_spineshoulder]) - getVectorForm(_readFileClimberKinectInfo[_spinemid])).norm();
			spine_length = (getVectorForm(_readFileClimberKinectInfo[_spinemid]) - getVectorForm(_readFileClimberKinectInfo[_spinebase])).norm();

			thigh_length = ((getVectorForm(_readFileClimberKinectInfo[_hipright]) - getVectorForm(_readFileClimberKinectInfo[_kneeright])).norm()
							+ (getVectorForm(_readFileClimberKinectInfo[_hipleft]) - getVectorForm(_readFileClimberKinectInfo[_kneeleft])).norm()) / 2.0f;
			leg_length = ((getVectorForm(_readFileClimberKinectInfo[_kneeright]) - getVectorForm(_readFileClimberKinectInfo[_ankleright])).norm()
							+ (getVectorForm(_readFileClimberKinectInfo[_kneeleft]) - getVectorForm(_readFileClimberKinectInfo[_ankleleft])).norm()) / 2.0f;
			//feet_length = 2.5f * (((getVectorForm(_readFileClimberKinectInfo[_groundfeetright]) - getVectorForm(_readFileClimberKinectInfo[_feetright])).norm()
						  //	+ (getVectorForm(_readFileClimberKinectInfo[_groundfeetleft]) - getVectorForm(_readFileClimberKinectInfo[_feetleft])).norm()) / 2.0f);

			shoulder_length = ((getVectorForm(_readFileClimberKinectInfo[_shoulderright]) - getVectorForm(_readFileClimberKinectInfo[_elbowright])).norm()
							+ (getVectorForm(_readFileClimberKinectInfo[_shoulderleft]) - getVectorForm(_readFileClimberKinectInfo[_elbowleft])).norm()) / 2.0f;
			arm_length = ((getVectorForm(_readFileClimberKinectInfo[_elbowright]) - getVectorForm(_readFileClimberKinectInfo[_wristright])).norm()
							+ (getVectorForm(_readFileClimberKinectInfo[_elbowleft]) - getVectorForm(_readFileClimberKinectInfo[_wristleft])).norm()) / 2.0f;
			//hand_length = ((getVectorForm(_readFileClimberKinectInfo[_wristright]) - getVectorForm(_readFileClimberKinectInfo[_handtipright])).norm()
			//				+ (getVectorForm(_readFileClimberKinectInfo[_wristleft]) - getVectorForm(_readFileClimberKinectInfo[_handtipleft])).norm()) / 2.0f;

			midPos = (getVectorForm(_readFileClimberKinectInfo[_shoulderright]) + getVectorForm(_readFileClimberKinectInfo[_shoulderleft])) / 2.0f;
			dis_spine_shoulder_z = -(midPos[1] - _readFileClimberKinectInfo[_spineshoulder][1]);
			dis_spine_shoulder_x = (getVectorForm(_readFileClimberKinectInfo[_shoulderright]) - getVectorForm(_readFileClimberKinectInfo[_shoulderleft])).norm() / 2.0f;

			midPos = (getVectorForm(_readFileClimberKinectInfo[_hipright]) + getVectorForm(_readFileClimberKinectInfo[_hipleft])) / 2.0f;
			dis_spine_leg_z = _readFileClimberKinectInfo[_spinebase][1] - midPos[1];
			dis_spine_leg_x = (getVectorForm(_readFileClimberKinectInfo[_hipright]) - getVectorForm(_readFileClimberKinectInfo[_hipleft])).norm() / 2.0f;

			head_length = (2.0f) * (getVectorForm(_readFileClimberKinectInfo[_head]) - getVectorForm(_readFileClimberKinectInfo[_neck])).norm(); 
			dis_spine_head_z = (getVectorForm(_readFileClimberKinectInfo[_neck]) - getVectorForm(_readFileClimberKinectInfo[_spineshoulder])).norm();

			feetWidth = ((getVectorForm(_readFileClimberKinectInfo[_groundfeetright]) - getVectorForm(_readFileClimberKinectInfo[_ankleright])).norm()
							+ (getVectorForm(_readFileClimberKinectInfo[_groundfeetleft]) - getVectorForm(_readFileClimberKinectInfo[_ankleleft])).norm()) / 2.0f;
		}

		climberRadius = trunk_length + spine_length + leg_length + thigh_length + shoulder_length + arm_length;
		climberLegLegDis = 2 * (leg_length + thigh_length);
		climberHandHandDis = 2 * (shoulder_length + arm_length);
		climberHeight = feetWidth + leg_length + thigh_length + dis_spine_leg_z + spine_length + trunk_length + dis_spine_head_z + head_length;

		float pZ = spine_length + thigh_length + dis_spine_leg_z + leg_length + ArmLegWidth;

		float trunkPosX = pX;
		float trunkPosY = pY;
		float trunkPosZ = pZ;
		int cJointID = -1;

		// trunk (root body without parent)
		createBodyi(BodyName::BodyTrunk, (1.2f * boneRadius) * scale, trunk_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyTrunk], pX, pY, pZ + trunk_length / 2);
		fatherBodyIDs[BodyName::BodyTrunk] = -1;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyTrunk)], unsigned long(0x0080));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyTrunk)], unsigned long(0x36FE));

		// spine
		pZ -= (spine_length / 2);
		createBodyi(BodyName::BodySpine, (boneRadius) * scale, spine_length);
		odeBodySetPosition(bodyIDs[BodyName::BodySpine], pX, pY, pZ);
		createJointType(pX, pY, pZ + (spine_length / 2), BodyName::BodyTrunk, BodyName::BodySpine);
		cJointID = createJointType(pX, pY, pZ + (spine_length / 2), BodyName::BodyTrunk, BodyName::BodySpine, JointType::ThreeDegreeJoint);
		std::vector<Vector2> cJointLimits = setAngleLimitations(cJointID, BodyName::BodySpine);
		setJointID(BodyName::BodySpine, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodySpine] = BodyName::BodyTrunk;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodySpine)], unsigned long(0x0001));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodySpine)], unsigned long(0x7F6D));

		// left thigh
		pZ = trunkPosZ;
		pX -= (dis_spine_leg_x); 
		pZ -= (spine_length + thigh_length / 2 + dis_spine_leg_z); 
		createBodyi(BodyName::BodyLeftThigh, ArmLegWidth, thigh_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyLeftThigh], pX, pY, pZ);
		createJointType(pX, pY, pZ + (thigh_length / 2), BodyName::BodySpine, BodyName::BodyLeftThigh);
		cJointID = createJointType(pX, pY, pZ + (thigh_length / 2), BodyName::BodySpine, BodyName::BodyLeftThigh, JointType::ThreeDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyLeftThigh);
		setJointID(BodyName::BodyLeftThigh, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyLeftThigh] = BodyName::BodyTrunkLower;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyLeftThigh)], unsigned long(0x0010));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyLeftThigh)], unsigned long(0x7FDE));

		// left leg
		pZ -= (thigh_length / 2 + leg_length / 2);
		createBodyi(BodyName::BodyLeftLeg, ArmLegWidth, leg_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyLeftLeg], pX, pY, pZ);
		cJointID = createJointType(pX, pY, pZ + (leg_length / 2), BodyName::BodyLeftThigh, BodyName::BodyLeftLeg, JointType::OneDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyLeftLeg);
		setJointID(BodyName::BodyLeftLeg, cJointID, JointType::OneDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyLeftLeg] = BodyName::BodyLeftThigh;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyLeftLeg)], unsigned long(0x0020));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyLeftLeg)], unsigned long(0x7FAF));

		// left foot end point
		pZ -= (leg_length / 2 + ArmLegWidth / 4.0f);
		pY += (feet_length / 2 - ArmLegWidth / 2.0f);
		createBodyi(BodyName::BodyLeftFoot, ArmLegWidth * 0.9f, feet_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyLeftFoot], pX, pY, pZ);
		cJointID = createJointType(pX, trunkPosY, pZ + ArmLegWidth / 4.0f, BodyName::BodyLeftLeg, BodyName::BodyLeftFoot, JointType::OneDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyLeftFoot);
		setJointID(BodyName::BodyLeftFoot, cJointID, JointType::OneDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyLeftFoot] = BodyName::BodyLeftLeg;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyLeftFoot)], unsigned long(0x0040));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyLeftFoot)], unsigned long(0x7FDF));

		// right thigh
		pX = trunkPosX;
		pY = trunkPosY;
		pZ = trunkPosZ;

		pX += (dis_spine_leg_x);
		pZ -= (spine_length + thigh_length / 2 + dis_spine_leg_z);
		createBodyi(BodyName::BodyRightThigh, ArmLegWidth, thigh_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyRightThigh], pX, pY, pZ);
		createJointType(pX, pY, pZ + (thigh_length / 2.0f), BodyName::BodySpine, BodyName::BodyRightThigh);
		cJointID = createJointType(pX, pY, pZ + (thigh_length / 2.0f), BodyName::BodySpine, BodyName::BodyRightThigh, JointType::ThreeDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyRightThigh);
		setJointID(BodyName::BodyRightThigh, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyRightThigh] = BodyName::BodyTrunkLower;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyRightThigh)], unsigned long(0x0002));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyRightThigh)], unsigned long(0x7FFA));

		// right leg
		pZ -= (thigh_length / 2 + leg_length / 2);
		createBodyi(BodyName::BodyRightLeg, ArmLegWidth, leg_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyRightLeg], pX, pY, pZ);
		cJointID = createJointType(pX, pY, pZ + (leg_length / 2.0f), BodyName::BodyRightThigh, BodyName::BodyRightLeg, JointType::OneDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyRightLeg);
		setJointID(BodyName::BodyRightLeg, cJointID, JointType::OneDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyRightLeg] = BodyName::BodyRightThigh;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyRightLeg)], unsigned long(0x0004));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyRightLeg)], unsigned long(0x7FF5));

		// right foot end point
		pZ -= (leg_length / 2 + ArmLegWidth / 4.0f);
		pY += (feet_length / 2 - ArmLegWidth / 2);
		createBodyi(BodyName::BodyRightFoot, ArmLegWidth * 0.9f, feet_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyRightFoot], pX, pY, pZ);
		cJointID = createJointType(pX, trunkPosY, pZ + ArmLegWidth / 4.0f, BodyName::BodyRightLeg, BodyName::BodyRightFoot, JointType::OneDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyRightFoot);
		setJointID(BodyName::BodyRightFoot, cJointID, JointType::OneDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyRightFoot] = BodyName::BodyRightLeg;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyRightFoot)], unsigned long(0x0008));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyRightFoot)], unsigned long(0x7FFB));

		// left shoulder
		pX = trunkPosX;
		pY = trunkPosY;
		pZ = trunkPosZ;

		pX -= (shoulder_length / 2.0f + dis_spine_shoulder_x);
		pZ += (trunk_length - dis_spine_shoulder_z);
		createBodyi(BodyName::BodyLeftShoulder, shoulder_length, handsWidth);
		odeBodySetPosition(bodyIDs[BodyName::BodyLeftShoulder], pX, pY, pZ);
		createJointType(pX + (shoulder_length / 2.0f), pY, pZ, BodyName::BodyTrunk, BodyName::BodyLeftShoulder);
		cJointID = createJointType(pX + (shoulder_length / 2.0f), pY, pZ, BodyName::BodyTrunk, BodyName::BodyLeftShoulder, JointType::ThreeDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyLeftShoulder);
		setJointID(BodyName::BodyLeftShoulder, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyLeftShoulder] = BodyName::BodyTrunkUpper;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyLeftShoulder)], unsigned long(0x0800));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyLeftShoulder)], unsigned long(0x6F7F));

		// left arm
		pX -= (shoulder_length / 2 + arm_length / 2);
		createBodyi(BodyName::BodyLeftArm, arm_length, handsWidth);
		odeBodySetPosition(bodyIDs[BodyName::BodyLeftArm], pX, pY, pZ);
		cJointID = createJointType(pX + (arm_length / 2.0f), pY, pZ, BodyName::BodyLeftShoulder, BodyName::BodyLeftArm, JointType::OneDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyLeftArm);
		setJointID(BodyName::BodyLeftArm, cJointID, JointType::OneDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyLeftArm] = BodyName::BodyLeftShoulder;

		odeGeomSetCategoryBits(mGeomID[BodyName::BodyLeftArm], unsigned long(0x1000));
		odeGeomSetCollideBits (mGeomID[BodyName::BodyLeftArm], unsigned long(0x57FF));

		// left hand end point
		pX -= (arm_length / 2 + (hand_length / 2.0f));
		createBodyi(BodyName::BodyLeftHand, hand_length, handsWidth);
		odeBodySetPosition(bodyIDs[BodyName::BodyLeftHand], pX, pY, pZ);
		createJointType(pX + (hand_length / 2.0f), pY, pZ, BodyName::BodyLeftArm, BodyName::BodyLeftHand);
		cJointID = createJointType(pX + (hand_length / 2.0f), pY, pZ, BodyName::BodyLeftArm, BodyName::BodyLeftHand, JointType::ThreeDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyLeftHand);
		setJointID(BodyName::BodyLeftHand, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyLeftHand] = BodyName::BodyLeftArm;

		odeGeomSetCategoryBits(mGeomID[BodyName::BodyLeftHand], unsigned long(0x2000));
		odeGeomSetCollideBits (mGeomID[BodyName::BodyLeftHand], unsigned long(0x6FFF));

		// right shoulder
		pX = trunkPosX;
		pZ = trunkPosZ;

		pX += (shoulder_length / 2.0f + dis_spine_shoulder_x);
		pZ += (trunk_length - dis_spine_shoulder_z);
		createBodyi(BodyName::BodyRightShoulder, shoulder_length, handsWidth);
		odeBodySetPosition(bodyIDs[BodyName::BodyRightShoulder], pX, pY, pZ);
		createJointType(pX - (shoulder_length / 2.0f), pY, pZ, BodyName::BodyTrunk, BodyName::BodyRightShoulder);
		cJointID = createJointType(pX - (shoulder_length / 2.0f), pY, pZ, BodyName::BodyTrunk, BodyName::BodyRightShoulder, JointType::ThreeDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyRightShoulder);
		setJointID(BodyName::BodyRightShoulder, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyRightShoulder] = BodyName::BodyTrunkUpper;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyRightShoulder)], unsigned long(0x0100));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyRightShoulder)], unsigned long(0x7D7F));

		// right arm
		pX += (shoulder_length / 2 + arm_length / 2);
		createBodyi(BodyName::BodyRightArm, arm_length, handsWidth);
		odeBodySetPosition(bodyIDs[BodyName::BodyRightArm], pX, pY, pZ);
		cJointID = createJointType(pX - (arm_length / 2.0f), pY, pZ, BodyName::BodyRightShoulder, BodyName::BodyRightArm, JointType::OneDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyRightArm);
		setJointID(BodyName::BodyRightArm, cJointID, JointType::OneDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyRightArm] = BodyName::BodyRightShoulder;

		odeGeomSetCategoryBits(mGeomID[BodyName::BodyRightArm], unsigned long(0x0200));
		odeGeomSetCollideBits (mGeomID[BodyName::BodyRightArm], unsigned long(0x7AFF));

		// right hand end point
		pX += (arm_length / 2 + (hand_length / 2.0f));
		createBodyi(BodyName::BodyRightHand, hand_length, handsWidth);
		odeBodySetPosition(bodyIDs[BodyName::BodyRightHand], pX, pY, pZ);
		createJointType(pX - (hand_length / 2.0f), pY, pZ, BodyName::BodyRightArm, BodyName::BodyRightHand);
		cJointID = createJointType(pX - (hand_length / 2.0f), pY, pZ, BodyName::BodyRightArm, BodyName::BodyRightHand, JointType::ThreeDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyRightHand);
		setJointID(BodyName::BodyRightHand, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyRightHand] = BodyName::BodyRightArm;

		odeGeomSetCategoryBits(mGeomID[BodyName::BodyRightHand], unsigned long(0x0400));
		odeGeomSetCollideBits (mGeomID[BodyName::BodyRightHand], unsigned long(0x7DFF));

		// head
		pX = trunkPosX;
		pZ = trunkPosZ;

		pZ += (trunk_length + head_length / 2 + dis_spine_head_z);
		createBodyi(BodyName::BodyHead, HeadWidth, head_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyHead], pX, pY, pZ);
		createJointType(pX, pY, pZ - (head_length / 2.0f), BodyName::BodyTrunk, BodyName::BodyHead);
		cJointID = createJointType(pX, pY, pZ - (head_length / 2.0f), BodyName::BodyTrunk, BodyName::BodyHead, JointType::ThreeDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyHead);
		setJointID(BodyName::BodyHead, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyHead] = BodyName::BodyTrunkUpper;

		odeGeomSetCategoryBits(mGeomID[BodyName::BodyHead], unsigned long(0x4000));
		odeGeomSetCollideBits (mGeomID[BodyName::BodyHead], unsigned long(0x7F7F));

		float mass = 0;
		for (int i = 0; i < BodyNUM; i++)
		{
			mass += odeBodyGetMass(bodyIDs[i]);
		}
		float scaleFactor = dmass / mass;
		for (int i = 0; i < BodyNUM; i++)
		{
			float mass_i = odeBodyGetMass(bodyIDs[i]);
			float length,radius;
			odeGeomCapsuleGetParams(mGeomID[i],radius,length);
			odeMassSetCapsuleTotal(bodyIDs[i],mass_i*scaleFactor,radius,length);
		}

		return;
	}

	void setJointID(BodyName iBodyName, int iJointID, int iJointType, std::vector<Vector2>& iJointLimits)
	{
		jointIDs[iBodyName - 1] = iJointID;
		jointTypes[iBodyName - 1] = iJointType;

		if (iJointType == JointType::ThreeDegreeJoint)
		{
			jointIDIndex.push_back(iBodyName - 1);
			jointIDIndex.push_back(iBodyName - 1);
			jointIDIndex.push_back(iBodyName - 1);

			jointAxisIndex.push_back(0);
			jointAxisIndex.push_back(1);
			jointAxisIndex.push_back(2);

			jointLimits.push_back(iJointLimits[0]);
			jointLimits.push_back(iJointLimits[1]);
			jointLimits.push_back(iJointLimits[2]);
		}
		else
		{
			jointIDIndex.push_back(iBodyName - 1);
			jointAxisIndex.push_back(0);
			jointLimits.push_back(iJointLimits[0]);
		}
	}
	
	std::vector<int> createBodyi(int i, float lx, float lz, BodyType bodyType = BodyType::BodyCapsule)
	{
		int bodyID = odeBodyCreate();
		if (i < BodyNUM && i >= 0)
		{
			bodyIDs[i] = bodyID;
			bodyTypes[i] = bodyType;
		}

		float m_body_size = lz;
		float m_body_width = lx;
		if (i == BodyName::BodyLeftArm || i == BodyName::BodyLeftShoulder || i == BodyName::BodyRightArm || i == BodyName::BodyRightShoulder 
			|| i == BodyName::BodyRightHand || i == BodyName::BodyLeftHand)
		{
			m_body_size = lx;
			m_body_width = lz;
		}
		if (i < BodyNUM && i >= 0)
		{
			boneSize[i] = m_body_size;
		}

		int cGeomID = -1;

		if (bodyType == BodyType::BodyBox)
		{
			odeMassSetBox(bodyID, DENSITY, m_body_width, boneRadius, m_body_size);
			cGeomID = odeCreateBox(m_body_width, boneRadius, m_body_size);
		}
		else if (bodyType == BodyType::BodyCapsule)
		{
			m_body_width *= 0.5f;
			
			if (i == BodyName::BodyLeftArm || i == BodyName::BodyLeftShoulder || i == BodyName::BodyRightArm || i == BodyName::BodyRightShoulder 
				|| i == BodyName::BodyRightHand || i == BodyName::BodyLeftHand)
			{
				dMatrix3 R;
				dRFromAxisAndAngle(R, 0, 1, 0, PI /2);
				odeBodySetRotation(bodyID, R);
			}
			if (i == BodyName::BodyRightFoot || i == BodyName::BodyLeftFoot)
			{
				dMatrix3 R;
				dRFromAxisAndAngle(R, 1, 0, 0, PI /2);
				odeBodySetRotation(bodyID, R);
			}
			odeMassSetCapsule(bodyID, DENSITY, m_body_width, m_body_size);
			cGeomID = odeCreateCapsule(0, m_body_width, m_body_size);
		}
		else
		{
			m_body_width *= 0.5;
			odeMassSetSphere(bodyID, DENSITY / 10, m_body_width);
			cGeomID = odeCreateSphere(m_body_width);
		}

		if (i < BodyNUM && i >= 0)
		{
			mGeomID[i] = cGeomID;
		}

		if (i >= 0)
		{
			odeGeomSetCollideBits (cGeomID, 1 << (i + 1)); 
			odeGeomSetCategoryBits (cGeomID, 1 << (i + 1)); 
		}

		odeGeomSetBody(cGeomID, bodyID);

		std::vector<int> ret_val;
		ret_val.push_back(bodyID);
		ret_val.push_back(cGeomID);
		if (i < BodyNUM && i >= 0)
		{
			initialRotations[i]=ode2eigenq(odeBodyGetQuaternion(bodyID));
		}

		return ret_val;
	}

	std::vector<Vector2> setAngleLimitations(int jointID, BodyName iBodyName)
	{
		float hipSwingFwd = convertToRad(130.0f);
        float hipSwingBack = convertToRad(20.0f);
        float hipSwingOutwards = convertToRad(70.0f);
        float hipSwingInwards = convertToRad(15.0f);
        float hipTwistInwards = convertToRad(15.0f);
        float hipTwistOutwards = convertToRad(45.0f);
            
		float shoulderSwingFwd = convertToRad(160.0f);
        float shoulderSwingBack = convertToRad(20.0f);
        float shoulderSwingOutwards = convertToRad(30.0f);
        float shoulderSwingInwards = convertToRad(100.0f);
        float shoulderTwistUp = convertToRad(80.0f);		//in t-pose, think of bending elbow so that hand points forward. This twist direction makes the hand go up
        float shoulderTwistDown = convertToRad(20.0f);

		float spineSwingSideways = convertToRad(20.0f);
        float spineSwingForward = convertToRad(40.0f);
        float spineSwingBack = convertToRad(10.0f);
        float spineTwist = convertToRad(30.0f);
        

		float fwd_limit = 30.0f * (PI / 180.0f);
		float tilt_limit = 10.0f * (PI / 180.0f);
		float twist_limit = 45.0f * (PI / 180.0f);
		/*float wristSwingFwd = 15.0f;
        float wristSwingBack = 15.0f;
        float wristSwingOutwards = 70.0f;
        float wristSwingInwards = 15.0f;
        float wristTwistRange = 30.0f;
        float ankleSwingRange = 30.0f;
        float kneeSwingRange = 140.0f;*/

		std::vector<Vector2> cJointLimits;
		const float elbowStraightLimit=AaltoGames::deg2rad*1.0f;
		const float kneeStraightLimit=AaltoGames::deg2rad*2.0f;
		const float elbowKneeBentLimit=deg2rad*150.0f;
		switch (iBodyName)
		{
		case BodyName::BodySpine:
			odeJointSetAMotorParam(jointID, dParamLoStop1, -spineTwist); // x
			odeJointSetAMotorParam(jointID, dParamHiStop1, spineTwist);

			cJointLimits.push_back(Vector2(-spineTwist,spineTwist));

			odeJointSetAMotorParam(jointID, dParamLoStop2, -spineSwingSideways); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, spineSwingSideways);

			cJointLimits.push_back(Vector2(-spineSwingSideways,spineSwingSideways));

			odeJointSetAMotorParam(jointID, dParamLoStop3, -spineSwingForward); // z
			odeJointSetAMotorParam(jointID, dParamHiStop3, spineSwingBack);

			cJointLimits.push_back(Vector2(-spineSwingForward,spineSwingBack));
			break;
		case BodyName::BodyLeftThigh:
			odeJointSetAMotorParam(jointID, dParamLoStop1, -hipSwingOutwards); // z
			odeJointSetAMotorParam(jointID, dParamHiStop1, hipSwingInwards);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop1),odeJointGetAMotorParam(jointID,dParamHiStop1)));

			odeJointSetAMotorParam(jointID, dParamLoStop2, -hipTwistOutwards); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, hipTwistInwards);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop2),odeJointGetAMotorParam(jointID,dParamHiStop2)));

			odeJointSetAMotorParam(jointID, dParamLoStop3, -hipSwingFwd); // x
			odeJointSetAMotorParam(jointID, dParamHiStop3, hipSwingBack);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop3),odeJointGetAMotorParam(jointID,dParamHiStop3)));
			break;
		case BodyName::BodyRightThigh:
			odeJointSetAMotorParam(jointID, dParamLoStop1, -hipSwingInwards); // z
			odeJointSetAMotorParam(jointID, dParamHiStop1, hipSwingOutwards);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop1),odeJointGetAMotorParam(jointID,dParamHiStop1)));

			odeJointSetAMotorParam(jointID, dParamLoStop2, -hipTwistInwards); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, hipTwistOutwards);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop2),odeJointGetAMotorParam(jointID,dParamHiStop2)));

			odeJointSetAMotorParam(jointID, dParamLoStop3, -hipSwingFwd); // x
			odeJointSetAMotorParam(jointID, dParamHiStop3, hipSwingBack);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop3),odeJointGetAMotorParam(jointID,dParamHiStop3)));
			break;
		case BodyName::BodyLeftShoulder:
			odeJointSetAMotorParam(jointID, dParamLoStop1, -shoulderSwingOutwards); // z
			odeJointSetAMotorParam(jointID, dParamHiStop1, shoulderSwingInwards);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop1),odeJointGetAMotorParam(jointID,dParamHiStop1)));

			odeJointSetAMotorParam(jointID, dParamLoStop2, -shoulderTwistDown); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, shoulderTwistUp);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop2),odeJointGetAMotorParam(jointID,dParamHiStop2)));

			odeJointSetAMotorParam(jointID, dParamLoStop3, -shoulderSwingBack); // x
			odeJointSetAMotorParam(jointID, dParamHiStop3, shoulderSwingFwd);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop3),odeJointGetAMotorParam(jointID,dParamHiStop3)));
			break;
		case BodyName::BodyRightShoulder:
			odeJointSetAMotorParam(jointID, dParamLoStop1, -shoulderSwingInwards); // z
			odeJointSetAMotorParam(jointID, dParamHiStop1, shoulderSwingOutwards);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop1),odeJointGetAMotorParam(jointID,dParamHiStop1)));

			odeJointSetAMotorParam(jointID, dParamLoStop2, -shoulderTwistDown); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, shoulderTwistUp);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop2),odeJointGetAMotorParam(jointID,dParamHiStop2)));

			odeJointSetAMotorParam(jointID, dParamLoStop3, -shoulderSwingFwd); // x
			odeJointSetAMotorParam(jointID, dParamHiStop3, shoulderSwingBack);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop3),odeJointGetAMotorParam(jointID,dParamHiStop3)));
			break;
		case BodyName::BodyLeftLeg:
			odeJointSetHingeParam(jointID, dParamLoStop, kneeStraightLimit);
			odeJointSetHingeParam(jointID, dParamHiStop, elbowKneeBentLimit);

			cJointLimits.push_back(Vector2(kneeStraightLimit,elbowKneeBentLimit));
			break;
		case BodyName::BodyRightLeg:
			odeJointSetHingeParam(jointID, dParamLoStop, kneeStraightLimit);
			odeJointSetHingeParam(jointID, dParamHiStop, elbowKneeBentLimit);

			cJointLimits.push_back(Vector2(kneeStraightLimit,elbowKneeBentLimit));
			break;
		case BodyName::BodyLeftArm:
			odeJointSetHingeParam(jointID, dParamLoStop, elbowStraightLimit);
			odeJointSetHingeParam(jointID, dParamHiStop, elbowKneeBentLimit);

			cJointLimits.push_back(Vector2(elbowStraightLimit,elbowKneeBentLimit));
			break;
		case BodyName::BodyRightArm:
			odeJointSetHingeParam(jointID, dParamLoStop, -elbowKneeBentLimit);
			odeJointSetHingeParam(jointID, dParamHiStop, -elbowStraightLimit);

			cJointLimits.push_back(Vector2(-elbowKneeBentLimit,-elbowStraightLimit));
			break;
		case BodyName::BodyHead:
			

			odeJointSetAMotorParam(jointID, dParamLoStop1, -fwd_limit); // x
			odeJointSetAMotorParam(jointID, dParamHiStop1, fwd_limit);

			cJointLimits.push_back(Vector2(-fwd_limit,fwd_limit));

			

			odeJointSetAMotorParam(jointID, dParamLoStop2, -tilt_limit); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, tilt_limit);

			cJointLimits.push_back(Vector2(-tilt_limit,tilt_limit));

			

			odeJointSetAMotorParam(jointID, dParamLoStop3, -twist_limit); // z
			odeJointSetAMotorParam(jointID, dParamHiStop3, twist_limit);

			cJointLimits.push_back(Vector2(-twist_limit,twist_limit));
			break;
		case BodyName::BodyRightFoot:
			odeJointSetHingeParam(jointID, dParamLoStop, -15 * (PI / 180.0f));
			odeJointSetHingeParam(jointID, dParamHiStop, 45 * (PI / 180.0f));

			//odeJointSetAMotorParam(jointID, dParamLoStop1, -PI / 4); // x
			//odeJointSetAMotorParam(jointID, dParamHiStop1, PI / 4);

			//cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			//odeJointSetAMotorParam(jointID, dParamLoStop2, -PI / 4); // y
			//odeJointSetAMotorParam(jointID, dParamHiStop2, PI / 4);

			//cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			//odeJointSetAMotorParam(jointID, dParamLoStop3, -PI / 4); // z
			//odeJointSetAMotorParam(jointID, dParamHiStop3, PI / 4);

			cJointLimits.push_back(Vector2(-15 * (PI / 180.0f),45 * (PI / 180.0f)));
			break;
		case BodyName::BodyLeftHand:
			odeJointSetAMotorParam(jointID, dParamLoStop1, -PI / 4); // x
			odeJointSetAMotorParam(jointID, dParamHiStop1, PI / 4);

			cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			odeJointSetAMotorParam(jointID, dParamLoStop2, -PI / 4); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, PI / 4);

			cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			odeJointSetAMotorParam(jointID, dParamLoStop3, -PI / 4); // z
			odeJointSetAMotorParam(jointID, dParamHiStop3, PI / 4);

			cJointLimits.push_back(Vector2(-PI / 4,PI / 4));
			break;
		case BodyName::BodyRightHand:
			odeJointSetAMotorParam(jointID, dParamLoStop1, -PI / 4); // x
			odeJointSetAMotorParam(jointID, dParamHiStop1, PI / 4);

			cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			odeJointSetAMotorParam(jointID, dParamLoStop2, -PI / 4); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, PI / 4);

			cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			odeJointSetAMotorParam(jointID, dParamLoStop3, -PI / 4); // z
			odeJointSetAMotorParam(jointID, dParamHiStop3, PI / 4);

			cJointLimits.push_back(Vector2(-PI / 4,PI / 4));
			break;
		case BodyName::BodyLeftFoot:
			odeJointSetHingeParam(jointID, dParamLoStop, -15 * (PI / 180.0f));
			odeJointSetHingeParam(jointID, dParamHiStop, 45 * (PI / 180.0f));
			//odeJointSetAMotorParam(jointID, dParamLoStop1, -PI / 4); // x
			//odeJointSetAMotorParam(jointID, dParamHiStop1, PI / 4);

			//cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			//odeJointSetAMotorParam(jointID, dParamLoStop2, -PI / 4); // y
			//odeJointSetAMotorParam(jointID, dParamHiStop2, PI / 4);

			//cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			//odeJointSetAMotorParam(jointID, dParamLoStop3, -PI / 4); // z
			//odeJointSetAMotorParam(jointID, dParamHiStop3, PI / 4);

			cJointLimits.push_back(Vector2(-15 * (PI / 180.0f),45 * (PI / 180.0f)));
			break;
		default:
			odeJointSetAMotorParam(jointID, dParamLoStop2, -PI / 2); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, PI / 2);
		}

		return cJointLimits;
	}

	float convertToRad(float iDegree)
	{
		return iDegree * (PI / 180.0f);
	}

	//////////////////////////////////////////// variables /////////////////////////////////////////////

	Vector3 goal_pos;

	// for drawing of Env bodies
	int startingHoldGeomsIndex;
	std::vector<int> mENVGeoms;
	std::vector<int> mENVGeomTypes;

	// variables for creating joint in hold positions
	std::vector<int> jointHoldBallIDs; // only 4 ball joint IDs

	// variable refers to hold pos index
	std::vector<std::vector<int>> holdPosIndex; // only 4 hold IDs for each context

	// for drawing humanoid body
	std::vector<int> bodyIDs;
	std::vector<int> fatherBodyIDs;
	std::vector<int> mGeomID; 
	std::vector<int> bodyTypes;

	// variable for getting end point of the body i
	std::vector<float> boneSize;
	
	std::vector<int> jointIDs;
	std::vector<int> jointTypes;
	int mJointSize;
	std::vector<int> jointIDIndex;
	std::vector<int> jointAxisIndex;
	std::vector<Vector2> jointLimits;

	// for testing the humanoid climber
	std::vector<float> desiredAnglesBones; 

	int masterContext;
	int maxNumContexts; 
	int currentFreeSavingStateSlot;

	// climber's info
	float climberRadius; // maximum dis between hand and foot
	float climberLegLegDis; // maximum dis between leg to leg
	float climberHandHandDis; // masimum dis between hand to hand
	float climberHeight; // climber's height from its feet to head

}* mContext;

#include <deque>
#include <future>
class mController 
{
public:
	mController()
	{
		counter_let_go = std::vector<std::vector<int>>(4, std::vector<int>(contextNUM, 0));
		keepBothFeetUp = std::vector<bool>(contextNUM, false);
	}

	std::vector<std::vector<BipedState>> states_trajectory_steps;

	enum ControlledPoses { MiddleTrunk = 0, LeftLeg = 1, RightLeg = 2, LeftHand = 3, RightHand = 4, Posture = 5, TorsoDir = 6 };
	enum StanceBodies {sbLeftLeg=0,sbRightLeg,sbLeftHand,sbRightHand};
	enum CostComponentNames {VioateDis, Velocity, ViolateVel, Angles, cLeftLeg, cRightLeg, cLeftHand, cRightHand, cMiddleTrunkP, cMiddleTrunkD, cHead, cNaN};

	int masterContextID;

	BipedState startState;
	BipedState resetState;
	float current_cost_state;
	float current_cost_control;

	virtual void optimize_the_cost(bool advance_time, std::vector<ControlledPoses>& sourcePos, std::vector<Vector3>& targetPos, std::vector<int>& targetHoldIDs, bool showDebugInfo) = 0;

	virtual void syncMasterContextWithStartState(bool loadAnyWay) = 0;

	virtual bool simulateBestTrajectory(bool flagSaveSlots, std::vector<int>& dHoldIDs, std::vector<BipedState>& outStates) = 0;

	virtual void reset() = 0;

	void visualizeForceDirections(bool printDebugInfo = true)
	{
		rcSetColor(1,0,0);

		Vector3 fHand = startState.forces[0];
		if (printDebugInfo) rcPrintString("force on left leg, x:%f, y:%f, z:%f", fHand.x(), fHand.y(), fHand.z());
		Vector3 p1 = mContextController->getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
		Vector3 p2 = p1 + fHand.normalized();
		mContextController->drawLine(p1, p2);

		fHand = startState.forces[1];
		if (printDebugInfo) rcPrintString("force on right leg, x:%f, y:%f, z:%f", fHand.x(), fHand.y(), fHand.z());
		p1 = mContextController->getEndPointPosBones(SimulationContext::BodyName::BodyRightLeg);
		p2 = p1 + fHand.normalized();
		mContextController->drawLine(p1, p2);

		fHand = startState.forces[2];
		if (printDebugInfo) rcPrintString("force on left hand, x:%f, y:%f, z:%f", fHand.x(), fHand.y(), fHand.z());
		p1 = mContextController->getEndPointPosBones(SimulationContext::BodyName::BodyLeftHand);
		p2 = p1 + fHand.normalized();
		mContextController->drawLine(p1, p2);

		fHand = startState.forces[3];
		if (printDebugInfo) rcPrintString("force on right hand, x:%f, y:%f, z:%f", fHand.x(), fHand.y(), fHand.z());
		p1 = mContextController->getEndPointPosBones(SimulationContext::BodyName::BodyRightHand);
		p2 = p1 + fHand.normalized();
		mContextController->drawLine(p1, p2);
		return;
	}

protected:
	Eigen::VectorXf control_init_tmp;
	Eigen::VectorXf controlMin;
	Eigen::VectorXf controlMax;
	Eigen::VectorXf controlMean;
	Eigen::VectorXf controlSd;
	Eigen::VectorXf poseMin;
	Eigen::VectorXf poseMax;
	Eigen::VectorXf defaultPose;

	std::vector<Eigen::VectorXf> stateFeatures;

	std::vector<VectorXf> posePriorSd, posePriorMean, threadControls;

	SimulationContext* mContextController;

	std::vector<std::vector<int>> counter_let_go;
	std::vector<bool> keepBothFeetUp;

	// same functions
	virtual Eigen::VectorXf getBestControl(int cTimeStep) = 0;

	virtual void apply_control(SimulationContext* iContextCPBP, const Eigen::VectorXf& control) = 0;
	
	// assumes we are in the "trajectory_idx" context, and save or restore should be done after this function based on returned "physicsBroken"
	virtual bool advance_simulation_context(Eigen::VectorXf& cControl, int trajectory_idx, float& controlCost, std::vector<int>& dHoldIDs,
		bool debugPrint, bool flagSaveSlots, std::vector<BipedState>& nStates) = 0;

	int getRandomIndex(unsigned int iArraySize)
	{
		if (iArraySize == 0)
			return -1;
		int m_index = rand() % iArraySize;

		return m_index;
	}

	// assumes you are in correct simulation context and state
	void saveBipedStateSimulationTrajectory(int trajectory_idx, int cStep)
	{
		saveOdeState(states_trajectory_steps[trajectory_idx][cStep].saving_slot_state, trajectory_idx);
		states_trajectory_steps[trajectory_idx][cStep].hold_bodies_ids = mContextController->holdPosIndex[trajectory_idx];

		for (int i = 0; i < 4; i++)
		{
			states_trajectory_steps[trajectory_idx][cStep].forces[i] = 
				mContextController->getForceVectorOnEndBody((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), trajectory_idx);
		}
		
		return;
	}

	///////////// implemetation of linear force model //////////////////////////////////
	float calculateLinearForceModel(SimulationContext* iContextCont, int holdNum, Vector3& cFi, int targetContext)
	{
		float f_max = 0.0f;
		if (iContextCont->getHoldBodyIDs(holdNum, targetContext) >= 0)
		{
			float f_ideal = iContextCont->holds_body[iContextCont->getHoldBodyIDs(holdNum, targetContext)].f_ideal;
			Vector3 d_ideal = iContextCont->holds_body[iContextCont->getHoldBodyIDs(holdNum, targetContext)].d_ideal;
			float k = iContextCont->holds_body[iContextCont->getHoldBodyIDs(holdNum, targetContext)].k;
			f_max = f_ideal * max<float>(0, 1 - k * SimulationContext::getAbsAngleBtwVectors(cFi, d_ideal));
		}
		return f_max;
	}
	///////////////////////*********************************////////////////////////////

	//////////// connect or disconnect hands and feet of the climber ///////////////////
	void mConnectDisconnectContactPoint(std::vector<int>& desired_holds_ids, int targetContext, int max_leg_go, std::vector<bool>& _allowRelease)
	{
		float min_reject_angle = (PI / 2) - (0.3f * PI);
		for (unsigned int i = 0; i < desired_holds_ids.size(); i++)
		{
			SimulationContext::ContactPoints mContactPoint = (SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i);

			Vector3 fi = mContextController->getForceVectorOnEndBody(mContactPoint, targetContext);

			if (i <= 1) // left leg and right leg
			{
				//////////////////////////////////////////// falling feature /////////////////////////////////////////////////////////////
				if (mContextController->getHoldBodyIDs(2, targetContext) == -1 && mContextController->getHoldBodyIDs(3, targetContext) == -1)
				{					
					if (fi.norm() > 0.0f)
					{
						mContextController->detachContactPoint(mContactPoint, targetContext);	
					}
					continue;
				}
			}

			float f_max = calculateLinearForceModel(mContextController, i, fi, targetContext);
			if (desired_holds_ids[i] != -1)
			{
				Vector3 hold_pos_i = mContextController->getHoldPos(desired_holds_ids[i]);
				Vector3 contact_pos_i = mContextController->getEndPointPosBones(SimulationContext::ContactPoints::LeftLeg + i);
				float dis_i = (hold_pos_i - contact_pos_i).norm();

				float _connectionThreshold = 0.24f * mContextController->getHoldSize(desired_holds_ids[i]);

				if (dis_i <= _connectionThreshold)
				{
					//// considering the effect of direction of the force
					//if (fi.norm() > f_max)
					//{
					//	counter_let_go[i][targetContext]++;
					//}
					//else
					//{
					//	counter_let_go[i][targetContext] = counter_let_go[i][targetContext] > 0 ? counter_let_go[i][targetContext]-1: 0;
					//}
					//if (counter_let_go[i][targetContext] >= max_leg_go)
					//{
					//	mContextController->detachContactPoint(mContactPoint, targetContext);
					//	counter_let_go[i][targetContext] = 0;
					//	continue;
					//}

					if (i <= 1) // left leg and right leg
					{
						Vector3 dir_contact_pos = -mContextController->getBodyDirectionZ(SimulationContext::ContactPoints::LeftLeg + i);

						float m_angle_btw = SimulationContext::getAbsAngleBtwVectors(-Vector3(0.0f, 0.0f, 1.0f), dir_contact_pos);
						if (m_angle_btw > min_reject_angle)
						{
							if (desired_holds_ids[0] != desired_holds_ids[1])
							{
								mContextController->attachContactPointToHold(mContactPoint, desired_holds_ids[i], targetContext);
							}
							else if (mContextController->getHoldBodyIDs(0, targetContext) != desired_holds_ids[1] && mContextController->getHoldBodyIDs(1, targetContext) != desired_holds_ids[0])
							{
								mContextController->attachContactPointToHold(mContactPoint, desired_holds_ids[i], targetContext);
							}
						}
						else
						{
							mContextController->detachContactPoint(mContactPoint, targetContext);
						}
					}
					else
					{
						mContextController->attachContactPointToHold(mContactPoint, desired_holds_ids[i], targetContext);
					}
				}
				else if ((dis_i > _connectionThreshold + 0.1f || desired_holds_ids[i] != mContextController->getHoldBodyIDs(i, targetContext)) && _allowRelease[i])
				{
					mContextController->detachContactPoint(mContactPoint, targetContext);
				}
			}
			else
			{
				if (_allowRelease[i])
				{
					mContextController->detachContactPoint(mContactPoint, targetContext);
				}
			}
		}

		return;
	}

	// we want to copy from fcontext to tcontext
	bool disconnectHandsFeetJointsFromTo(int fContext, int tContext)
	{
		bool flag_set_state = false;
		for (int i = 0; i < mContextController->getHoldBodyIDsSize(); i++)
		{
			if (mContextController->getHoldBodyIDs(i, fContext) != mContextController->getHoldBodyIDs(i, tContext))
			{
				mContextController->detachContactPoint((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), tContext);
				flag_set_state = true;
			}
		}
		return flag_set_state;
	}
	// we want to copy from fcontext to tcontext
	void connectHandsFeetJointsFromTo(int fContext, int tContext)
	{
		for (unsigned int i = 0; i < startState.hold_bodies_ids.size(); i++)
		{
			int attachHoldId = mContextController->getHoldBodyIDs(i, fContext);
			if (attachHoldId != mContextController->getHoldBodyIDs(i, tContext))
			{
				mContextController->detachContactPoint((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), tContext);
				if (attachHoldId >= 0)
				{
					mContextController->attachContactPointToHold((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), attachHoldId, tContext);
				}
			}
		}
		return;
	}

	bool disconnectHandsFeetJoints(int targetContext)
	{
		bool flag_set_state = false;
		for (int i = 0; i < mContextController->getHoldBodyIDsSize(); i++)
		{
			if (targetContext == ALLTHREADS)
			{
				for (int c = 0; c < mContextController->maxNumContexts; c++)
				{
					if (startState.hold_bodies_ids[i] != mContextController->getHoldBodyIDs(i, c))
					{
						mContextController->detachContactPoint((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), c);
						flag_set_state = true;
					}
				}
			}
			else
			{
				if (startState.hold_bodies_ids[i] != mContextController->getHoldBodyIDs(i, targetContext))
				{
					mContextController->detachContactPoint((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), targetContext);
					flag_set_state = true;
				}
			}
		}
		return flag_set_state;
	}

	void connectHandsFeetJoints(int targetContext)
	{
		for (unsigned int i = 0; i < startState.hold_bodies_ids.size(); i++)
		{
			if (targetContext == ALLTHREADS)
			{
				for (int c = 0; c < mContextController->maxNumContexts; c++)
				{
					if (startState.hold_bodies_ids[i] != mContextController->getHoldBodyIDs(i, c))
					{
						mContextController->detachContactPoint((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), c);
						if (startState.hold_bodies_ids[i] >= 0)
						{
							mContextController->attachContactPointToHold((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), startState.hold_bodies_ids[i], c);
						}
					}
				}
			}
			else
			{
				if (startState.hold_bodies_ids[i] != mContextController->getHoldBodyIDs(i, targetContext))
				{
					mContextController->detachContactPoint((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), targetContext);
					if (startState.hold_bodies_ids[i] >= 0)
					{
						mContextController->attachContactPointToHold((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), startState.hold_bodies_ids[i], targetContext);
					}
				}
			}
		}
		return;
	}

	////////////////////////////////////////////// debug visualize for climber*s state ////////////////////////////////////////////////
	void debug_visualize(int step)
	{
		Vector3 red = Vector3(0, 0, 255.0f) / 255.0f;
		Vector3 green = Vector3(0, 255.0f, 0) / 255.0f;
		Vector3 cyan = Vector3(255.0f, 255.0f, 0) / 255.0f;

		for (int t = nTrajectories - 1; t >= 0; t--)
		{
			int idx = t;

			Vector3 color = Vector3(0.5f, 0.5f,0.5f);
			if (t==0)
				color=green;

				rcSetColor(color.x(), color.y(), color.z());
				SimulationContext::drawLine(mContextController->initialPosition[idx], mContextController->resultPosition[idx]);
		}

		return;
	}

	// assumes the correct context is resored
	void debugVisulizeForceOnHnadsFeet(int targetContext, bool printDebugInfo = true)
	{
		rcSetColor(1,0,0);

		Vector3 fHand = mContextController->getForceVectorOnEndBody(SimulationContext::ContactPoints::LeftLeg, targetContext);
		if (printDebugInfo) rcPrintString("force on left leg, x:%f, y:%f, z:%f", fHand.x(), fHand.y(), fHand.z());
		Vector3 p1 = mContextController->getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
		Vector3 p2 = p1 + fHand.normalized();
		mContextController->drawLine(p1, p2);

		fHand = mContextController->getForceVectorOnEndBody(SimulationContext::ContactPoints::RightLeg, targetContext);
		if (printDebugInfo) rcPrintString("force on right leg, x:%f, y:%f, z:%f", fHand.x(), fHand.y(), fHand.z());
		p1 = mContextController->getEndPointPosBones(SimulationContext::BodyName::BodyRightLeg);
		p2 = p1 + fHand.normalized();
		mContextController->drawLine(p1, p2);

		fHand = mContextController->getForceVectorOnEndBody(SimulationContext::ContactPoints::LeftArm, targetContext);
		if (printDebugInfo) rcPrintString("force on left hand, x:%f, y:%f, z:%f", fHand.x(), fHand.y(), fHand.z());
		p1 = mContextController->getEndPointPosBones(SimulationContext::BodyName::BodyLeftHand);
		p2 = p1 + fHand.normalized();
		mContextController->drawLine(p1, p2);

		fHand = mContextController->getForceVectorOnEndBody(SimulationContext::ContactPoints::RightArm, targetContext);
		if (printDebugInfo) rcPrintString("force on right hand, x:%f, y:%f, z:%f", fHand.x(), fHand.y(), fHand.z());
		p1 = mContextController->getEndPointPosBones(SimulationContext::BodyName::BodyRightHand);
		p2 = p1 + fHand.normalized();
		mContextController->drawLine(p1, p2);

		return;
	}

	////////////////////////////////////////////// compute control and state costs ////////////////////////////////////////////////////
	static float compute_control_cost(SimulationContext* iContextCPBP, int targetContext)
	{
		float result=0;
		for (int j = 0; j < iContextCPBP->getJointSize(); j++)
		{
			result += iContextCPBP->getMotorAppliedSqTorque(j)/squared(forceCostSd);
		}
		result+=iContextCPBP->getSqForceOnFingers(targetContext)/squared(forceCostSd);
		

		return result;
	}

	float computeStateCost(SimulationContext* iContextCPBP, std::vector<ControlledPoses>& sourcePos, std::vector<Vector3>& targetPos, std::vector<int>& targetHoldIDs
		,  bool printDebug, int targetContext)
	{
		// we assume that we are in the correct simulation context using restoreOdeState
		float trunkDistSd = 100.0f*0.2f;  //nowadays this is only based on distance from wall
		float angleSd = poseAngleSd;  //loose prior, seems to make more natural movement
		float tPoseAngleSd = deg2rad*5.0f;

		float endEffectorDistSd = optimizerType == otCMAES ? 0.0025f : 0.0025f;  //CMAES only evaluates hold targets at last frame, need a larger weight
		float velSd = optimizerType == otCMAES ? 100.0f : 0.15f;  //velSd only needed for C-PBP to reduce noise
		float chestDirSd = optimizerType == otCMAES ? 0.05f : 0.05f;

		float forceSd = optimizerType == otCMAES ? 50.0f : 200.0f;
		float otherPoseDir = 0.1f;
		float forcePoseDir = 0.5f;
		float COMDirSd = optimizerType == otCMAES ? 0.1f : 0.075f;

		float stateCost = 0;
		float preStateCost = 0;
		
		bool includeTargetHoldCost = false;
		int counterContactPoint = 0;
		for (unsigned int i = 0; i < targetHoldIDs.size(); i++)
		{
			if (targetHoldIDs[i] != iContextCPBP->getHoldBodyIDs(i, targetContext))
			{
				includeTargetHoldCost = true;
			}
			if (targetHoldIDs[i] > -1)
			{
				counterContactPoint++;
			}
		}

		stateCost += ((Vector3(0.0f,0.0f,1.0f) - iContextCPBP->getBodyDirectionZ(SimulationContext::BodyName::BodyTrunk)) * (1 / otherPoseDir)).squaredNorm();
		Vector3 ll = iContextCPBP->getEndPointPosBones(SimulationContext::ContactPoints::LeftLeg);
		Vector3 rl = iContextCPBP->getEndPointPosBones(SimulationContext::ContactPoints::RightLeg);
		Vector3 trunkPos = iContextCPBP->getEndPointPosBones(SimulationContext::BodyName::BodyTrunk);
		float _standing_dis = FLT_MAX;
		if (iContextCPBP->getHoldBodyIDs(0, targetContext) >= 0) // if ll connected
		{
			_standing_dis = min(_standing_dis,(ll-trunkPos).norm());
		}
		if (iContextCPBP->getHoldBodyIDs(1, targetContext) >= 0) // if rl connected
		{
			_standing_dis = min(_standing_dis,(rl-trunkPos).norm());
		}
		if (_standing_dis < 1.0f)
		{
			stateCost += squared((1 / _standing_dis) * (1 / otherPoseDir));
		}

		if (printDebug)
		{
			rcPrintString("Standing dis cost %f",stateCost - preStateCost);
			preStateCost = stateCost;
		}

		if (iContextCPBP->getHoldBodyIDs(2, targetContext) >= 0 || iContextCPBP->getHoldBodyIDs(3, targetContext) >= 0) // if one hand connected
		{
			rl = iContextCPBP->getEndPointPosBones(SimulationContext::ContactPoints::RightLeg, true);
			ll = iContextCPBP->getEndPointPosBones(SimulationContext::ContactPoints::LeftLeg, true);
			// find the other leg
			if (!keepBothFeetUp[targetContext] || targetHoldIDs[0] >= 0 || targetHoldIDs[1] >= 0)
			{
				if (targetHoldIDs[0] >= 0 && targetHoldIDs[1] == -1)
				{
					if (rl[2] < ll[2] + 0.1f && rl[2] < 0.5f)
						stateCost += squared(20 * (ll[2] + 0.1f - rl[2]) / otherPoseDir);
					keepBothFeetUp[targetContext] = true;
				}
				if (targetHoldIDs[1] >= 0 && targetHoldIDs[0] == -1)
				{
					if (ll[2] < rl[2] + 0.1f && ll[2] < 0.5f)
						stateCost += squared(20 * (rl[2] + 0.1f - ll[2]) / otherPoseDir);
					keepBothFeetUp[targetContext] = true;
				}
			}
			else
			{
				if (targetHoldIDs[0] == -1)
				{
					if (ll[2] < 0.5f)
						stateCost += squared(20 * (0.5f - ll[2]) / otherPoseDir);
				}
				if (targetHoldIDs[1] == -1)
				{
					if (rl[2] < 0.5f)
						stateCost += squared(20 * (0.5f - rl[2]) / otherPoseDir);
				}
			}
		}
		else
		{
			keepBothFeetUp[targetContext] = false;
		}

		if (printDebug)
		{
			rcPrintString("foot above ground cost %f",stateCost - preStateCost);
			preStateCost = stateCost;
		}

		for (unsigned int i = 0; i < targetHoldIDs.size(); i++)
		{
			int _hold_id_i = iContextCPBP->getHoldBodyIDs(i, targetContext);
			
			Vector3 dirBody = iContextCPBP->getBodyDirectionZ(SimulationContext::BodyName::BodyLeftLeg + i);
			if (i <= 1)
			{
				// keep the leg up right
				stateCost += ((Vector3(0.0f,0.0f,-1.0f) - dirBody) * (1 / otherPoseDir)).squaredNorm();
			}

			if (_hold_id_i >= 0)
			{
				if (i <= 1)
				{
					if (i == 0)
						dirBody = iContextCPBP->getBodyDirectionY(SimulationContext::BodyName::BodyLeftFoot);
					else
						dirBody = iContextCPBP->getBodyDirectionY(SimulationContext::BodyName::BodyRightFoot);
				}
				
				float res_pos_1 = ((iContextCPBP->holds_body[_hold_id_i].d_ideal - dirBody) * (1 / COMDirSd)).squaredNorm(); // pushing
				float res_pos_2 = ((-iContextCPBP->holds_body[_hold_id_i].d_ideal - dirBody) * (1 / COMDirSd)).squaredNorm(); // pulling

				float res_pos = min(res_pos_1, res_pos_2);
				if (i > 1)
				{
					switch (iContextCPBP->holds_body[_hold_id_i].HoldPushPullMode)
					{
					case 0:// pulling
						res_pos = res_pos_2;
						break;
					case 1:// pushing
						res_pos = res_pos_1;
						break;
					case 2:
						res_pos = min(res_pos_1, res_pos_2);
					}
				}
				stateCost += res_pos;
			}
		}
		for (unsigned int i = 0; i < targetHoldIDs.size(); i++)
		{
			if (iContextCPBP->getHoldBodyIDs(i, targetContext) != targetHoldIDs[i])
			{
				stateCost += squared(2 * (1 / COMDirSd));
				counter_let_go[i][targetContext] = 0;
			}
		}
		if (printDebug)
		{
			rcPrintString("Direction cost %f",stateCost - preStateCost);
			preStateCost = stateCost;
		}

		//Vector3 upperTrunkPos = iContextCPBP->getEndPointPosBones(SimulationContext::BodyName::BodyTrunkUpper);
		//for (unsigned int i = 0; i < targetHoldIDs.size(); i++)
		//{
		//	Vector3 fi = iContextCPBP->getForceVectorOnEndBody((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), targetContext);
		//	float f_max = calculateLinearForceModel(iContextCPBP, i, fi, targetContext);
		//	float cF = fi.norm();
		//	int _hold_id_i = iContextCPBP->getHoldBodyIDs(i, targetContext);
		//	//if (_hold_id_i >= 0 && cF > f_max)
		//	//{
		//	//	stateCost += squared(cF - f_max) / (forceSd * forceSd);
		//	//}
		//	// optimize the direction of force!
		//	if (_hold_id_i >= 0)
		//	{
		//		if (i > 1)
		//		{
		//			Vector3 d_dir = (upperTrunkPos - iContextCPBP->holds_body[_hold_id_i].holdPos).normalized();
		//			if (upperTrunkPos[2] > iContextCPBP->holds_body[_hold_id_i].holdPos[2])
		//			{
		//				d_dir = (iContextCPBP->holds_body[_hold_id_i].holdPos - upperTrunkPos).normalized();
		//			}
		//			stateCost += (d_dir - fi.normalized()).squaredNorm() / (forcePoseDir * forcePoseDir);
		//		}
		//		else
		//		{
		//			stateCost += (Vector3(0.0f,0.0f,-1.0f) - fi.normalized()).squaredNorm() / (forcePoseDir * forcePoseDir);
		//		}
		//	}
		//	else if (_hold_id_i != targetHoldIDs[i])
		//		stateCost += 4 / (forcePoseDir * forcePoseDir);
		//}
		//if (printDebug)
		//{
		//	rcPrintString("Force cost %f", stateCost - preStateCost);
		//	preStateCost = stateCost;
		//}

		////////////////////////////////////////////// Breaking Ode

		//for (int i = 0; i < BodyNUM; i++)
		//{
		//	stateCost += (iContextCPBP->getForceVectorOnEndBody(i, targetContext) / forceSd).squaredNorm();
		//}
		if (iContextCPBP->checkViolatingRelativeDis())
		{
			stateCost += 1e20;
		}

		if (printDebug)
		{
			rcPrintString("Distance violation cost %f",stateCost - preStateCost);
			preStateCost = stateCost;
		}

		/////////////////////////////////////////////// Velocity
		for (unsigned int k = 0; k < BodyNUM; k++)
		{
			stateCost += iContextCPBP->getBoneLinearVelocity(k).squaredNorm()/(velSd*velSd);
		}
		if (printDebug)
		{
			rcPrintString("Velocity cost %f",stateCost - preStateCost);
			preStateCost = stateCost;
		}

		///////////////////////////////////////////// Posture
		if (iContextCPBP->getHoldBodyIDs((int)(ControlledPoses::RightHand - ControlledPoses::LeftLeg), targetContext) != -1
			|| iContextCPBP->getHoldBodyIDs((int)(ControlledPoses::LeftHand - ControlledPoses::LeftLeg), targetContext) != -1)
		{
			if (optimizerType==otCMAES )  //in C-PBP, pose included in the sampling proposal. 
			{
				for (int k = 0; k < iContextCPBP->getJointSize(); k++)
				{
					float diff_ang = iContextCPBP->getDesMotorAngleFromID(k) - iContextCPBP->getJointAngle(k);

					stateCost += (squared(diff_ang) /(angleSd*angleSd));
				}
			}
		}
		else
		{
			for (unsigned int k = 0; k < iContextCPBP->bodyIDs.size(); k++)
			{
				if (k!=SimulationContext::BodyLeftShoulder
					&& k!=SimulationContext::BodyLeftArm
					&& k!=SimulationContext::BodyLeftHand
					&& k!=SimulationContext::BodyRightShoulder
					&& k!=SimulationContext::BodyRightArm
					&& k!=SimulationContext::BodyRightHand)
				{
					Eigen::Quaternionf q=ode2eigenq(odeBodyGetQuaternion(iContextCPBP->bodyIDs[k]));
					float diff=q.angularDistance(initialRotations[k]);
					stateCost += squared(diff) /squared(tPoseAngleSd);
				}
			}
		} 

		if (printDebug)
		{
			rcPrintString("Pose cost %f",stateCost - preStateCost);
			preStateCost = stateCost;
		}

		bool isTorsoDirAdded = false;
		for (unsigned int i = 0; i < sourcePos.size(); i++)
		{
			ControlledPoses posID = sourcePos[i];
			Vector3 dPos = targetPos[i];

			float cWeight = 1.0f;

			Vector3 cPos(0.0f, 0.0f, 0.0f);

			if ((int)posID <= ControlledPoses::RightHand && (int)posID >= ControlledPoses::LeftLeg)
			{
				cPos = iContextCPBP->getEndPointPosBones(SimulationContext::ContactPoints::LeftLeg + posID - ControlledPoses::LeftLeg);
				cWeight = 1.0f/endEffectorDistSd;//weight_very_important;

				if (printDebug)
				{
					rcSetColor(0, 0, 1);
					SimulationContext::drawCross(cPos);
				}

				int cHoldID = iContextCPBP->getHoldBodyIDs((int)(posID - ControlledPoses::LeftLeg), targetContext);
				if (cHoldID == targetHoldIDs[(int)(posID - ControlledPoses::LeftLeg)])// ((cPos - dPos).norm() < 0.25f * holdSize)//
				{
					cPos = dPos;
				}

				// penetrating the wall
				if (cPos.y() >= 0.1f)
				{
					stateCost = 1e20;
				}

				if (includeTargetHoldCost)
				{
					stateCost += (cWeight * (cPos - dPos)).squaredNorm();
				}
			}
			else if (posID == ControlledPoses::MiddleTrunk)
			{
				cPos = iContextCPBP->computeCOM();
				dPos[0] = cPos[0];//only distance from wall matters
				dPos[2] = cPos[2];

				cWeight = 1 / trunkDistSd; //weight_average_important;

				if (includeTargetHoldCost)
				{
					stateCost += (cWeight * (cPos - dPos)).squaredNorm();
				}
			}
			else if (posID == ControlledPoses::TorsoDir)
			{
				Vector3 dirTrunk = iContextCPBP->getBodyDirectionY(SimulationContext::BodyName::BodyTrunk);
				stateCost += ((dirTrunk - dPos)/chestDirSd).squaredNorm(); // chest toward desired direction specified by user
				isTorsoDirAdded = true;
			}

			if (printDebug)
			{
				if ((int)posID == ControlledPoses::LeftLeg)
				{
					rcPrintString("Left leg cost %f",stateCost - preStateCost);
				}
				else if ((int)posID == ControlledPoses::RightLeg)
				{
					rcPrintString("Right leg cost %f",stateCost - preStateCost);
				}
				else if ((int)posID == ControlledPoses::LeftHand)
				{
					rcPrintString("Left hand cost %f",stateCost - preStateCost);
				}
				else if ((int)posID == ControlledPoses::RightHand)
				{
					rcPrintString("Right hand cost %f",stateCost - preStateCost);
				}
				else if ((int)posID == ControlledPoses::MiddleTrunk)
				{
					rcPrintString("Torso pos cost %f",stateCost - preStateCost);
				}
				else if (posID == ControlledPoses::TorsoDir)
				{
					rcPrintString("Chest direction cost %f",stateCost - preStateCost);
					preStateCost = stateCost;
				}
				preStateCost = stateCost;
			}
		}

		if (!isTorsoDirAdded)
		{
			Vector3 dirTrunk = iContextCPBP->getBodyDirectionY(SimulationContext::BodyName::BodyTrunk);
			stateCost += ((dirTrunk - Vector3(0,1,0))/chestDirSd).squaredNorm(); // chest toward the wall
			if (printDebug)
			{
				rcPrintString("Chest direction cost %f",stateCost - preStateCost);
				preStateCost = stateCost;
			}
		}

		if (stateCost != stateCost)
		{
			stateCost = 1e20;
		}

		return stateCost;
	} 

	/////////////////////////////////// computing feature for climber's state /////////////////////////////////////////////////////////
	static void pushStateFeature(int &featureIdx, float *stateFeatures, const Vector3& v)
	{
		stateFeatures[featureIdx++] = v.x();
		stateFeatures[featureIdx++] = v.y();
		stateFeatures[featureIdx++] = v.z();
	}

	static void pushStateFeature(int &featureIdx, float *stateFeatures, const Vector4& v)
	{
		stateFeatures[featureIdx++] = v.x();
		stateFeatures[featureIdx++] = v.y();
		stateFeatures[featureIdx++] = v.z();
		stateFeatures[featureIdx++] = v.w();
	}

	static void pushStateFeature(int &featureIdx, float *stateFeatures, const float& f)
	{
		stateFeatures[featureIdx++] = f;
	}

	static int computeStateFeatures(SimulationContext* iContextCPBP, float *stateFeatures, int targetContext) // BipedState state
	{
	
		int featureIdx = 0;
		const int nStateBones=6;
		SimulationContext::BodyName stateBones[6]={
			SimulationContext::BodySpine,
			SimulationContext::BodyTrunk,
			SimulationContext::BodyRightArm,
			SimulationContext::BodyRightLeg,
			SimulationContext::BodyLeftArm,
			SimulationContext::BodyLeftLeg};


		for (int i = 0; i < nStateBones; i++)
		{
			pushStateFeature(featureIdx, stateFeatures, iContextCPBP->getBonePosition(stateBones[i]));
			pushStateFeature(featureIdx, stateFeatures, iContextCPBP->getBoneLinearVelocity(stateBones[i]));
			pushStateFeature(featureIdx, stateFeatures, iContextCPBP->getBoneAngle(stateBones[i]));
			pushStateFeature(featureIdx, stateFeatures, iContextCPBP->getBoneAngularVelocity(stateBones[i]));
		}

		for (int i = 0; i < iContextCPBP->getHoldBodyIDsSize(); i++)
		{
			if (iContextCPBP->getHoldBodyIDs(i, targetContext) != -1)
			{
				pushStateFeature(featureIdx, stateFeatures, 1.0f);
			}
			else
			{
				pushStateFeature(featureIdx, stateFeatures, 0.0f);
			}
		}

		/*for (int i = 0; i < iContextCPBP->getHoldBodyIDsSize(); i++)
		{
			Vector3 fi = iContextCPBP->getForceVectorOnEndBody((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), targetContext);
			float f_max = calculateLinearForceModel(mContextController, i, fi, targetContext);
			pushStateFeature(featureIdx, stateFeatures, f_max - fi.norm());
		}*/

		return featureIdx;
	}
} *mOptimizer, *mOptimizerOffline;

class mOptCPBP : public mController
{
private:
	class CPBPTrajectoryResults
	{
	public:
		std::vector<VectorXf> control;
		int nSteps;
		float cost;
	};

	ControlPBP flc;
	Eigen::VectorXf controlDiffSd;
	Eigen::VectorXf controlDiffDiffSd;
	int currentIndexOfflineState;

	float bestCostOffLineCPBP;
	CPBPTrajectoryResults CPBPOfflineControls;
public:

	mOptCPBP(SimulationContext* iContexts, BipedState& iStartState, bool isOnline)
	{
		controlDiffSdScale = isOnline ? 0.5f : 0.2f;
		nPhysicsPerStep = optimizerType == otCMAES ? 1 : (isOnline ? 3 : 1);
		nTimeSteps = isOnline ? int(cTime*1.5001f) : int(cTime/2);

		int maxTimeSteps = nTimeSteps + 10;
		states_trajectory_steps = std::vector<std::vector<BipedState>>(nTrajectories + 1, std::vector<BipedState>(maxTimeSteps, iStartState));

		for (int i = 0; i <= nTrajectories; i++)
		{
			for (int j = 0; j < maxTimeSteps; j++)
			{
				states_trajectory_steps[i][j] = iStartState.getNewCopy(iContexts->getNextFreeSavingSlot(), iContexts->getMasterContextID());
			}
		}

		startState = iStartState;
		iContexts->saveContextIn(startState);

		startState.getNewCopy(iContexts->getNextFreeSavingSlot(), iContexts->getMasterContextID());

		current_cost_state = 0.0f;
		current_cost_control = 0.0f;

		mContextController = iContexts;
		masterContextID = mContextController->getMasterContextID();

		int control_size = mContextController->getJointSize() + fmCount + 4;

		control_init_tmp = Eigen::VectorXf::Zero(control_size);
		//controller init (motor target velocities are the controlled variables, one per joint)
		controlMin = control_init_tmp;
		controlMax = control_init_tmp;
		controlMean = control_init_tmp;
		controlSd = control_init_tmp;
		controlDiffSd = control_init_tmp;
		controlDiffDiffSd = control_init_tmp;
		poseMin=control_init_tmp;
		poseMax=control_init_tmp;
		defaultPose=control_init_tmp;

		for (int i = 0; i < mContextController->getJointSize(); i++)
		{
			//we are making everything relative to the rotation ranges. getBodyi() could also return the controlled body (corresponding to a BodyName enum)
			defaultPose[i]=iContexts->getDesMotorAngleFromID(i);
			poseMin[i]=iContexts->getJointAngleMin(i);
			poseMax[i]=iContexts->getJointAngleMax(i);
			float angleRange=poseMax[i]-poseMin[i];
			controlMin[i] = -maxSpeedRelToRange*angleRange;
			controlMax[i] = maxSpeedRelToRange*angleRange;

			controlMean[i] = 0;
			controlSd[i] = 0.5f * controlMax[i];
			controlDiffSd[i] = controlDiffSdScale * controlMax[i];		//favor small accelerations
			controlDiffDiffSd[i] = 1000.0f; //NOP
		}
		//fmax control
		for (int i = mContextController->getJointSize(); i < mContextController->getJointSize()+fmCount; i++)
		{
			//we are making everything relative to the rotation ranges. getBodyi() could also return the controlled body (corresponding to a BodyName enum)
			controlMin[i] = minimumForce;
			controlMax[i] = maximumForce;

			controlMean[i] = 0;
			controlSd[i] = controlMax[i];
			controlDiffSd[i] = controlDiffSdScale * controlMax[i];		//favor small accelerations
			controlDiffDiffSd[i] = 1000.0f; //NOP
		}
		//letting go control
		for (int i = mContextController->getJointSize()+fmCount; i < control_size; i++)
		{
			controlMin[i] = 0;
			controlMax[i] = 1;

			controlMean[i] = (controlMin[i] + controlMax[i]) / 2.0f;
			controlSd[i] = controlMax[i];
			controlDiffSd[i] = isOnline ? 0.2f * controlDiffSdScale * controlMax[i] : 0.6f * controlDiffSdScale * controlMax[i];		//favor small accelerations
			controlDiffDiffSd[i] = 1000.0f; //NOP
		}

		setCurrentOdeContext(masterContextID);

		float temp[1000];
		int stateDim = computeStateFeatures(mContextController, temp, masterContextID); // compute state features of startState or loaded master state
		Eigen::VectorXf stateSd(stateDim);
		for (int i = 0; i < stateDim; i++)
			stateSd[i] = 0.25f;
		float control_variation = 0.1f;

		flc.init(nTrajectories, nTimeSteps / nPhysicsPerStep, stateDim, control_size, controlMin.data()
			, controlMax.data(), controlMean.data(), controlSd.data(), controlDiffSd.data(), controlDiffDiffSd.data(), control_variation, NULL);
		flc.setParams(0.25f, 0.5f, false, 0.001f);

		int nPoseParams = mContextController->getJointSize();
		for (int i = 0; i <= nTrajectories; i++)
		{
			stateFeatures.push_back(Eigen::VectorXf::Zero(stateDim));
		}

		posePriorSd = std::vector<VectorXf>(nTrajectories, VectorXf(control_size));
		posePriorMean = std::vector<VectorXf>(nTrajectories, VectorXf(control_size));
		threadControls = std::vector<VectorXf>(nTrajectories, VectorXf(control_size));

		currentIndexOfflineState = 0;

		bestCostOffLineCPBP = FLT_MAX;
		CPBPOfflineControls.control = std::vector<VectorXf>(int(nTimeSteps / nPhysicsPerStep), VectorXf(control_size));
	}
	
	void reset()
	{
		flc.reset();

		bestCostOffLineCPBP = FLT_MAX;
		return;
	}

	void optimize_the_cost(bool advance_time, std::vector<ControlledPoses>& sourcePos, std::vector<Vector3>& targetPos, std::vector<int>& targetHoldIDs, bool showDebugInfo)
	{
		setCurrentOdeContext(masterContextID);

		int cContext = getCurrentOdeContext();

		restoreOdeState(masterContextID); // we have loaded master context state

        //Update the current state and pass it to the optimizer
		Eigen::VectorXf &stateFeatureMaster = stateFeatures[masterContextID];
		computeStateFeatures(mContextController, &stateFeatureMaster[0], masterContextID);
		
		if (advance_time)
		{
			//debug-print current state cost components
			float cStateCost = computeStateCost(mContextController, sourcePos, targetPos, targetHoldIDs, showDebugInfo, masterContextID); // true
			rcPrintString("Traj. cost for controller: %f", cStateCost);

			if (showDebugInfo) debugVisulizeForceOnHnadsFeet(masterContextID);
		}
		
		flc.startIteration(advance_time, &stateFeatureMaster[0]);
		bool standing = mContextController->getHoldBodyIDs((int)(sbLeftLeg), masterContextID)==-1 && mContextController->getHoldBodyIDs((int)(sbRightLeg), masterContextID)==-1;

		std::vector<int> nSteps(nTrajectories, 0);
		std::vector<std::vector<int>> bTrajectoryEachStep(nTimeSteps/nPhysicsPerStep, std::vector<int>(nTrajectories,-1));

		for (int step = 0; step < nTimeSteps/nPhysicsPerStep; step++)
		{
			flc.startPlanningStep(step);

			for (int i = 0; i < nTrajectories; i++)
			{
				if (step == 0)
                {
                    //save the physics state: at first step, the master context is copied to every other context
                    saveOdeState(i, masterContextID);
                }
                else
				{
                    //at others than the first step, just save each context so that the resampling can branch the paths
                    saveOdeState(i, i);
					bTrajectoryEachStep[step][i] = flc.getPreviousSampleIdx(i);
				}
			}
			
			std::deque<std::future<bool>> worker_queue;
			SimulationContext::BodyName targetDrawnLines = SimulationContext::BodyName::BodyTrunk;
			std::vector<BipedState> nStates;
			for (int t = nTrajectories - 1; t >= 0; t--)
			{
				//lambda to be executed in the thread of the simulation context
				auto simulate_one_step = [&](int trajectory_idx)
				{
					int previousStateIdx = flc.getPreviousSampleIdx(trajectory_idx);
					setCurrentOdeContext(trajectory_idx);

					int cContext = getCurrentOdeContext();
					
					disconnectHandsFeetJointsFromTo(previousStateIdx, trajectory_idx);
					restoreOdeState(previousStateIdx);
					connectHandsFeetJointsFromTo(previousStateIdx, trajectory_idx);

					//compute pose prior, needed for getControl()
					int nPoseParams = mContextController->getJointSize();
					int control_size = nPoseParams + fmCount + 4;

					float dt=timeStep*(float)nPhysicsPerStep;
					posePriorMean[trajectory_idx].setZero();
					if (!standing)
					{
						for (int i = 0; i < nPoseParams; i++)
						{
							posePriorMean[trajectory_idx][i] = (defaultPose[i]-mContextController->getJointAngle(i))/dt;
						}
					}
					else
					{
						for (int i = 0; i < nPoseParams; i++)
						{
							posePriorMean[trajectory_idx][i] = (0-mContextController->getJointAngle(i))/dt; //t-pose: zero angles
						}
					}
					posePriorSd[trajectory_idx].head(nPoseParams) = (poseMax.head(nPoseParams)-poseMin.head(nPoseParams))*(poseAngleSd/dt);
					posePriorSd[trajectory_idx].tail(control_size - nPoseParams).setConstant(1000.0f); //no additional prior on FMax //
					Eigen::VectorXf &control = threadControls[trajectory_idx]; 

					flc.getControl(trajectory_idx, control.data(),posePriorMean[trajectory_idx].data(),posePriorSd[trajectory_idx].data());
					

					//step physics
					mContextController->initialPosition[trajectory_idx] = mContextController->getEndPointPosBones(targetDrawnLines);
	
					//apply the random control and step forward and evaluate control cost 
					float controlCost = 0.0f;
					bool physicsBroken = false;
					for (int k = 0; k < nPhysicsPerStep && !physicsBroken; k++)
					{
						std::vector<bool> mAllowRelease;
						mAllowRelease.reserve(4);
						
//						if (!advance_time)
//						{   // for offline mode
						for (int _m = 0; _m < 4; _m++)
							mAllowRelease.push_back(0.5f > control[nPoseParams + fmCount + _m]); 
						physicsBroken = !advance_simulation_context(control, trajectory_idx, controlCost, targetHoldIDs, false, false, nStates);
//						}
						//else
						//{
						//	for (int _m = 0; _m < 4; _m++)
						//		mAllowRelease.push_back(true);
						//	physicsBroken = !advance_simulation_context(control, trajectory_idx, controlCost, targetHoldIDs, false, false, nStates, mAllowRelease);
						//}
						saveBipedStateSimulationTrajectory(trajectory_idx, nSteps[trajectory_idx]);

						nSteps[trajectory_idx] = nSteps[trajectory_idx] + 1;
					}

					float stateCost = 0;
					if (physicsBroken)
					{
						stateCost = 1e20;
						disconnectHandsFeetJointsFromTo(previousStateIdx, trajectory_idx);
						restoreOdeState(previousStateIdx);
						connectHandsFeetJointsFromTo(previousStateIdx, trajectory_idx);
					}
					else
					{
						//compute state cost, only including the hold costs at the last step
						stateCost = computeStateCost(mContextController, sourcePos, targetPos, targetHoldIDs, false, trajectory_idx); 
					}

					mContextController->resultPosition[trajectory_idx] = mContextController->getEndPointPosBones(targetDrawnLines);

					Eigen::VectorXf &stateFeatureOthers = stateFeatures[trajectory_idx];
					computeStateFeatures(mContextController, stateFeatureOthers.data(), trajectory_idx);
					//we can add control cost to state cost, as state cost is actually uniquely associated with a tuple [previous state, control, next state]
					flc.updateResults(trajectory_idx, control.data(), stateFeatureOthers.data(), stateCost+controlCost,posePriorMean[trajectory_idx].data(),posePriorSd[trajectory_idx].data());
					
					return true;
				};
		
				worker_queue.push_back(std::async(std::launch::async,simulate_one_step,t));
			}
		
			for (std::future<bool>& is_ready : worker_queue)
			{
				is_ready.wait();
			}

			flc.endPlanningStep(step);
			
			//debug visualization
			debug_visualize(step);
		}
		flc.endIteration();
		cContext = getCurrentOdeContext();

		current_cost_state = flc.getBestTrajectoryCost();

		// for offline mode
		if (!advance_time)
		{
			current_cost_state = flc.getBestTrajectoryCost();
			if (current_cost_state < bestCostOffLineCPBP)
			{
				bestCostOffLineCPBP = current_cost_state;
				CPBPOfflineControls.cost = bestCostOffLineCPBP;
				

				int cBestIdx = flc.getBestSampleLastIdx();
				
				CPBPOfflineControls.nSteps = nSteps[cBestIdx];

				int cStep = CPBPOfflineControls.nSteps - 1;
				for (int step = nTimeSteps/nPhysicsPerStep - 1; step >= 0; step--)
				{
					setCurrentOdeContext(cBestIdx);
					for (int n = 0; n < nPhysicsPerStep; n++)
					{
						restoreOdeState(states_trajectory_steps[cBestIdx][cStep].saving_slot_state);
						saveOdeState(states_trajectory_steps[nTrajectories][cStep].saving_slot_state, cBestIdx);

						states_trajectory_steps[nTrajectories][cStep].hold_bodies_ids = states_trajectory_steps[cBestIdx][cStep].hold_bodies_ids;
						for (int i = 0; i < 4; i++)
							states_trajectory_steps[nTrajectories][cStep].forces[i] = states_trajectory_steps[cBestIdx][cStep].forces[i];
						cStep--;
					}
					cBestIdx = bTrajectoryEachStep[step][cBestIdx];
				}

				for (int step = 0; step < nTimeSteps/nPhysicsPerStep; step++)
				{
					Eigen::VectorXf control = control_init_tmp;
					flc.getBestControl(step, control.data());
					CPBPOfflineControls.control[step] = control;
				}
			}

			//visualize
			setCurrentOdeContext(flc.getBestSampleLastIdx());
			mContextController->mDrawStuff(-1,-1,flc.getBestSampleLastIdx(), true, false);
			float stateCost = computeStateCost(mContextController, sourcePos, targetPos, targetHoldIDs, showDebugInfo, flc.getBestSampleLastIdx()); 
			rcPrintString("Traj. cost for controller: %f", current_cost_state);
			if (showDebugInfo) debugVisulizeForceOnHnadsFeet(flc.getBestSampleLastIdx());

			setCurrentOdeContext(masterContextID);
		}

		return;
	}
	
	// return true if simulation of the best trajectory is done from zero until maxTimeSteps otherwise false
	bool simulateBestTrajectory(bool flagSaveSlots, std::vector<int>& dHoldIDs, std::vector<BipedState>& outStates)
	{
		if (!useOfflinePlanning)
		{
			syncMasterContextWithStartState(true);

			setCurrentOdeContext(masterContextID);
			restoreOdeState(masterContextID);

			current_cost_control = 0.0f;
			bool physicsBroken = false;

			for (int k = 0; k < nPhysicsPerStep && !physicsBroken; k++)
			{
				std::vector<BipedState> nStates;
				float cControl = 0.0f;
				physicsBroken = !advance_simulation_context(getBestControl(0), // current best control
															masterContextID, // apply to master context
															cControl, // out current cost
															dHoldIDs, // desired hold ids
															false, // show debud info
															flagSaveSlots, // flag for saving immediate states
															nStates); // output states

				current_cost_control += cControl;

				mContextController->saveContextIn(startState);

				saveOdeState(masterContextID, masterContextID);
				
				for (unsigned int i = 0; i < nStates.size(); i++)
				{
					outStates.push_back(nStates[i]);
				}
			}

//			Sleep(30);
			
			startState.hold_bodies_ids = mContextController->holdPosIndex[masterContextID];

			
			if (!physicsBroken)
			{
				return true;
			}

			restoreOdeState(masterContextID);
			return false;
		}
		else
		{
			setCurrentOdeContext(masterContextID);

			int maxTimeSteps = CPBPOfflineControls.nSteps;
			if (currentIndexOfflineState < maxTimeSteps)
			{
				startState = states_trajectory_steps[nTrajectories][currentIndexOfflineState];
				syncMasterContextWithStartState(true);

				currentIndexOfflineState++;

//				Sleep(60);
				if (flagSaveSlots)
					outStates.push_back(this->startState.getNewCopy(mContextController->getNextFreeSavingSlot(), this->masterContextID));

				return false;
			}
			else
			{
				saveOdeState(masterContextID, masterContextID);

				currentIndexOfflineState = 0;

				return true;
			}
		}
	}

	// sync all contexts with the beginning state of the optimization
	void syncMasterContextWithStartState(bool loadAnyWay)
	{
		int cOdeContext = getCurrentOdeContext();

		setCurrentOdeContext(ALLTHREADS);

		bool flag_set_state = disconnectHandsFeetJoints(ALLTHREADS);

		bool flag_is_state_sync = false;
		// startState.saving_slot_state determines the current state that we want to be in
		if (masterContextID != startState.saving_slot_state || loadAnyWay)
		{
			restoreOdeState(startState.saving_slot_state, false);
			saveOdeState(masterContextID,0);
			mContextController->saveContextIn(startState);
			startState.saving_slot_state = masterContextID;
			flag_is_state_sync = true;
		}

		if (flag_set_state && !flag_is_state_sync)
		{
			restoreOdeState(masterContextID, false);
			saveOdeState(masterContextID,0);
			mContextController->saveContextIn(startState);
			flag_is_state_sync = true;
		}
		
		connectHandsFeetJoints(ALLTHREADS);

		setCurrentOdeContext(cOdeContext);

		return;
	}

private:
	// advance one step simulation assuming we are in the correct context then apply control and step, calculate the control cost, also return next simulated state if asked
	bool advance_simulation_context(Eigen::VectorXf& cControl, int trajectory_idx, float& controlCost, std::vector<int>& dHoldIds,
		bool debugPrint, bool flagSaveSlots, std::vector<BipedState>& nStates)
	{
		int max_let_go = 50;
		bool physicsBroken = false;

		int nPoseParams = mContextController->getJointSize();
		std::vector<bool> mAllowRelease;
		mAllowRelease.reserve(4);		
		for (int _m = 0; _m < 4; _m++)
			mAllowRelease.push_back(0.5f > cControl[nPoseParams + fmCount + _m]); 

		mConnectDisconnectContactPoint(dHoldIds, trajectory_idx, max_let_go, mAllowRelease);

		apply_control(mContextController, cControl);

		physicsBroken = !stepOde(timeStep,false);

		if (flagSaveSlots && !physicsBroken)
		{
			BipedState nState = this->startState.getNewCopy(mContextController->getNextFreeSavingSlot(), trajectory_idx);
			nStates.push_back(nState);
		}

		if (!physicsBroken)
		{
			controlCost += compute_control_cost(mContextController, trajectory_idx);
		}

		if (debugPrint) rcPrintString("Control cost for controller: %f",controlCost);

		if (!physicsBroken)
		{
			return true;
		}

		return false;
	}

	Eigen::VectorXf getBestControl(int cTimeStep)
	{
		if (!useOfflinePlanning)
		{
			Eigen::VectorXf control = control_init_tmp;
			flc.getBestControl(cTimeStep, control.data());
			return control;
		}
		else
		{
			return CPBPOfflineControls.control[cTimeStep];
		}
	}

	void apply_control(SimulationContext* iContextCPBP, const Eigen::VectorXf& control)
	{
		for (int j = 0; j < iContextCPBP->getJointSize(); j++)
		{
			float c_j = control[j];
			iContextCPBP->setMotorSpeed(j, c_j);
		}
		if (fmCount!=0)
		{
			//fmax values for each group are after the joint motor speeds
			iContextCPBP->setMotorGroupFmaxes(&control[iContextCPBP->getJointSize()]);
		}
	}

};

class mOptCMAES : public mController
{
private:
	class CMAESTrajectoryResults
	{
	public:
		static const int maxTimeSteps=512;
		VectorXf control[maxTimeSteps];
		int nSteps;
		float cost;
	};

	CMAES<float> cmaes;

	std::vector<std::vector<RecursiveTCBSpline>> splines;  //vector for each simulation context, each vector with the number of control params (joint velocities/poses and fmax)

	CMAESTrajectoryResults cmaesResults[nTrajectories];
	CMAESTrajectoryResults bestCmaesTrajectory;
	std::vector<std::pair<VectorXf,float> > cmaesSamples;

	int currentIndexOfflineState;
		
public:

	mOptCMAES(SimulationContext* iContexts, BipedState& iStartState)
	{
		int maxTimeSteps = int(maxFullTrajectoryDuration / timeStep) + 10;
		states_trajectory_steps = std::vector<std::vector<BipedState>>(nTrajectories + 1, std::vector<BipedState>(maxTimeSteps, iStartState));

		for (int i = 0; i <= nTrajectories; i++)
		{
			for (int j = 0; j < maxTimeSteps; j++)
			{
				states_trajectory_steps[i][j] = iStartState.getNewCopy(iContexts->getNextFreeSavingSlot(), iContexts->getMasterContextID());
			}
		}

		startState = iStartState;
		iContexts->saveContextIn(startState);

		startState.getNewCopy(iContexts->getNextFreeSavingSlot(), iContexts->getMasterContextID());

		current_cost_state = 0.0f;
		current_cost_control = 0.0f;

		mContextController = iContexts;
		masterContextID = mContextController->getMasterContextID();

		control_init_tmp = Eigen::VectorXf::Zero(mContextController->getJointSize() + fmCount);
		//controller init (motor target velocities are the controlled variables, one per joint)
		controlMin = control_init_tmp;
		controlMax = control_init_tmp;
		controlMean = control_init_tmp;
		controlSd = control_init_tmp;
		poseMin = control_init_tmp;
		poseMax = control_init_tmp;
		defaultPose = control_init_tmp;

		for (int i = 0; i < mContextController->getJointSize(); i++)
		{
			//we are making everything relative to the rotation ranges. getBodyi() could also return the controlled body (corresponding to a BodyName enum)
			defaultPose[i] = iContexts->getDesMotorAngleFromID(i);
			poseMin[i] = iContexts->getJointAngleMin(i);
			poseMax[i] = iContexts->getJointAngleMax(i);
			float angleRange = poseMax[i]-poseMin[i];
			controlMin[i] = -maxSpeedRelToRange*angleRange;
			controlMax[i] = maxSpeedRelToRange*angleRange;
			controlMean[i] = 0;
			controlSd[i] = 0.5f * controlMax[i];
		}

		//fmax control
		for (int i = mContextController->getJointSize(); i < mContextController->getJointSize()+fmCount; i++)
		{
			//we are making everything relative to the rotation ranges. getBodyi() could also return the controlled body (corresponding to a BodyName enum)
			controlMin[i] = minimumForce;
			controlMax[i] = maximumForce;
			controlMean[i] = 0;
			controlSd[i] = controlMax[i];
		}

		setCurrentOdeContext(masterContextID);

		float temp[1000];
		int stateDim = computeStateFeatures(mContextController, temp, masterContextID); // compute state features of startState or loaded master state
		Eigen::VectorXf stateSd(stateDim);
		for (int i = 0; i < stateDim; i++)
			stateSd[i] = 0.25f;
		float control_variation = 0.1f;

		//for each segment, the control vector has a duration value and a control point (motor speeds and fmaxes)
		cmaes.init_with_dimension(nCMAESSegments*(1+mContextController->getJointSize()+fmCount));
		cmaes.selected_samples_fraction_ = 0.5;
		cmaes.use_step_size_control_ = false ;
		cmaes.minimum_exploration_variance_=0.0005f;//*controlSd[0];
		splines.resize(contextNUM);
		for (int i=0; i<contextNUM; i++)
		{
			splines[i].resize(mContextController->getJointSize()+fmCount);
		}

		int nPoseParams = mContextController->getJointSize();
		for (int i = 0; i <= nTrajectories; i++)
		{
			stateFeatures.push_back(Eigen::VectorXf::Zero(stateDim));
		}

		posePriorSd = std::vector<VectorXf>(nTrajectories, VectorXf(nPoseParams+fmCount));
		posePriorMean = std::vector<VectorXf>(nTrajectories, VectorXf(nPoseParams+fmCount));
		threadControls = std::vector<VectorXf>(nTrajectories, VectorXf(nPoseParams+fmCount));

		currentIndexOfflineState = 0;
	}

	// return true if simulation of the best trajectory is done from zero until maxTimeSteps otherwise false
	bool simulateBestTrajectory(bool flagSaveSlots, std::vector<int>& dHoldIDs, std::vector<BipedState>& outStates)
	{
//		setCurrentOdeContext(masterContextID);
//		if (currentIndexOfflineState == 0)
//		{
//			syncMasterContextWithStartState();
//			restoreOdeState(masterContextID);
//		}
//
//		int maxTimeSteps = getNSteps();
//		if (currentIndexOfflineState <= maxTimeSteps)
//		{
//			std::vector<BipedState> nStates;
//			bool physicsBroken = !advance_simulation_context(
//				getBestControl(currentIndexOfflineState),
//				masterContextID, current_cost_control, false, flagSaveSlots, nStates);
//
//			if (!physicsBroken)
//			{
//				for (unsigned int i = 0; i < nStates.size(); i++)
//				{
//					outStates.push_back(nStates[i]);
//				}
//
//				currentIndexOfflineState++;
//
//				mContextController->saveContextIn(startState);
//
//				startState.hold_bodies_ids = mContextController->holdPosIndex[masterContextID];
//
//	//			saveOdeState(masterContextID, masterContextID);
//				Sleep(30);
//			}
//			else
//			{
////				restoreOdeState(masterContextID);
//			}
//			return false;
//		}
//		else
//		{
//			saveOdeState(masterContextID, masterContextID);
//
//			currentIndexOfflineState = 0;
//
//			return true;
//		}
		setCurrentOdeContext(masterContextID);

		int maxTimeSteps = getNSteps();
		if (currentIndexOfflineState < maxTimeSteps)
		{
			startState = states_trajectory_steps[nTrajectories][currentIndexOfflineState];
			syncMasterContextWithStartState();

			currentIndexOfflineState++;
			return false;
		}
		else
		{
			saveOdeState(masterContextID, masterContextID);

			currentIndexOfflineState = 0;

			return true;
		}

	}

	void optimize_the_cost(bool firstIter, std::vector<ControlledPoses>& sourcePos, std::vector<Vector3>& targetPos, std::vector<int>& targetHoldIDs, bool showDebugInfo)
	{
		setCurrentOdeContext(masterContextID);

		restoreOdeState(masterContextID); // we have loaded master context state
		
		int nVelocities = mContextController->getJointSize();
		int nValuesPerSegment = 1+nVelocities+fmCount;
		int nCurrentTrajectories = firstIter ? nTrajectories : nTrajectories / 2;
		cmaesSamples.resize(nCurrentTrajectories);

		//at first iteration, sample the population from the control prior
		if (firstIter)
		{
			bestCmaesTrajectory.cost = FLT_MAX;
			for (int sampleIdx=0; sampleIdx < nCurrentTrajectories; sampleIdx++)
			{
				VectorXf &sample = cmaesSamples[sampleIdx].first;
				sample.resize(nCMAESSegments*nValuesPerSegment);
				for (int segmentIdx=0; segmentIdx < nCMAESSegments; segmentIdx++)
				{
					int segmentStart=segmentIdx*nValuesPerSegment;
					sample[segmentStart] = minSegmentDuration+(maxSegmentDuration-minSegmentDuration)*randomf(); //duration in 0.2...1 seconds
					if (cmaesSamplePoses)
					{
						for (int velIdx=0; velIdx<nVelocities; velIdx++)
						{
							sample[segmentStart+1+velIdx] = randGaussianClipped(defaultPose[velIdx],poseAngleSd,poseMin[velIdx],poseMax[velIdx]);
						}
					}
					else
					{
						for (int velIdx = 0; velIdx < nVelocities; velIdx++)
						{
							sample[segmentStart+1+velIdx] = randGaussianClipped(controlMean[velIdx],controlSd[velIdx],controlMin[velIdx],controlMax[velIdx]);
						}
					}
					for (int fmaxIdx = 0; fmaxIdx < fmCount; fmaxIdx++)
					{
						//safest to start with high fmax and let the optimizer decrease them later, thus the mean at maximumForce
						sample[segmentStart+1+nVelocities+fmaxIdx] = randGaussianClipped(maximumForce,maximumForce-minimumForce,minimumForce,maximumForce);
					}
				}
			}
		}

		//at subsequent iterations, CMAES does the sampling. Note that we clamp the samples to bounds  
		else
		{
			cmaesSamples.resize(nCurrentTrajectories);
			auto points = cmaes.sample(nCurrentTrajectories);
			//cmaesSamples[0]=cmaesSamples[bestCmaEsTrajectoryIdx];
			for (int sampleIdx=0; sampleIdx<nCurrentTrajectories; sampleIdx++)
			{
				VectorXf &sample=cmaesSamples[sampleIdx].first;
				sample=points[sampleIdx];
				//clip the values
				for (int segmentIdx=0; segmentIdx<nCMAESSegments; segmentIdx++)
				{
					clampCMAESSegmentControls(segmentIdx,true,nVelocities,nValuesPerSegment,sample);
				}
			}
		}

		//evaluate the samples, i.e., simulate in parallel and accumulate cost
		std::deque<std::future<void>> worker_queue;
		std::vector<BipedState> nStates;

		counter_let_go = std::vector<std::vector<int>>(4, std::vector<int>(nTrajectories + 1, 0));
		std::vector<std::vector<int>> nLetGoStep(nTrajectories, std::vector<int>(4,0));
		/*for (int i = 0; i < nTrajectories; i++)
		{
			for (int j = 0; j < 4; j++)
				nLetGoStep[i][j] = 5 + getRandomIndex(0);
		}*/
		for (int sampleIdx = 0; sampleIdx < nCurrentTrajectories; sampleIdx++)
		{
			VectorXf &sample = cmaesSamples[sampleIdx].first;
			float &cost = cmaesSamples[sampleIdx].second;
			cost = 0;
			cmaesResults[sampleIdx].nSteps=0;
			//auto simulate_sample = [&sample,nValuesPerSegment](int sampleIdx)
			auto simulate_sample =[&](int trajectory_idx)
			{
				//restore physics state from master (all simulated trajectories start from current state)
				cmaesResults[trajectory_idx].nSteps=0;
				setCurrentOdeContext(trajectory_idx);

				disconnectHandsFeetJoints(trajectory_idx);
				restoreOdeState(masterContextID);
				connectHandsFeetJoints(trajectory_idx);
				
				mContextController->initialPosition[trajectory_idx] = mContextController->getEndPointPosBones(SimulationContext::BodyName::BodyTrunk);

				//setup spline interpolation initial state
				std::vector<RecursiveTCBSpline> &spl = splines[trajectory_idx]; //this context's spline interpolators
				for (int i=0; i<nVelocities; i++)
				{
					if (cmaesSamplePoses)
					{
						spl[i].setValueAndTangent(mContextController->getJointAngle(i),mContextController->getJointAngleRate(i));
						spl[i].linearMix = cmaesLinearInterpolation ? 1 : 0;
					}
					else
					{
						spl[i].setValueAndTangent(mContextController->getJointAngleRate(i),0);
						spl[i].linearMix = cmaesLinearInterpolation ? 1 : 0; 
					}
				}
				for (int i=nVelocities; i < nVelocities+fmCount; i++)
				{
					spl[i].setValueAndTangent(mContextController->getJointFMax(i-nVelocities),0);  
					spl[i].linearMix=cmaesLinearInterpolation ? 1 : 0;
				}

				//setup spline interpolation control points: we start at segment 0, and keep the times of the next two control points in t1 and t2
				int segmentIdx=0;
				float t1 = sample[segmentIdx*nValuesPerSegment];
				float t2 = t1 + sample[std::min(nCMAESSegments-1,segmentIdx+1)*nValuesPerSegment];
				int lastSegment = nCMAESSegments-1;
				if (!cmaesLinearInterpolation)
					lastSegment--; //in full spline interpolation, the last segment only defines the ending tangent
				float totalTime=0;

				std::vector<bool> mAllowedRelease;
				mAllowedRelease.reserve(4);
				for (int i=0; i<4; i++)
				{
					mAllowedRelease.push_back(cmaesResults[trajectory_idx].nSteps > nLetGoStep[trajectory_idx][i]);
				}
				mConnectDisconnectContactPoint(targetHoldIDs, trajectory_idx, 10000, mAllowedRelease);

				while (segmentIdx <= lastSegment && totalTime<maxFullTrajectoryDuration)
				{					
					//interpolate
					const VectorXf &controlPoint1 = sample.segment(segmentIdx*nValuesPerSegment+1,nVelocities+fmCount);
					const VectorXf &controlPoint2 = sample.segment(std::min(nCMAESSegments-1,segmentIdx+1)*nValuesPerSegment+1,nVelocities+fmCount);
					VectorXf &interpolatedControl = cmaesResults[trajectory_idx].control[cmaesResults[trajectory_idx].nSteps++];
					interpolatedControl.resize(nVelocities+fmCount);
					for (int i=0; i<nVelocities; i++)
					{
						float p1 = controlPoint1[i], p2 = controlPoint2[i];
						if (segmentIdx==lastSegment && forceCMAESLastValueToZero)
						{
							p1=0;
							p2=0;
						}
						spl[i].step(timeStep,p1,t1,p2,t2);
						interpolatedControl[i]=spl[i].getValue();
					}
					for (int i=nVelocities; i<nVelocities+fmCount; i++)
					{
						spl[i].step(timeStep, controlPoint1[i],t1,controlPoint2[i],t2);
						interpolatedControl[i]=spl[i].getValue();
					}

					//clamp for safety
					clampCMAESSegmentControls(0,false,nVelocities,nValuesPerSegment,interpolatedControl);

					//advance spline time, and start a new segment if needed
					t1 -= timeStep;
					t2 -= timeStep;
					totalTime+=timeStep;
					if (t1<0.001f)
					{
						segmentIdx++;
						t1 = sample[std::min(nCMAESSegments-1,segmentIdx)*nValuesPerSegment];
						t2 = t1 + sample[std::min(nCMAESSegments-1,segmentIdx+1)*nValuesPerSegment];
					}

					//apply the interpolated control and step forward and evaluate control cost 
					float controlCost = 0.0f;
					bool physicsBroken = !advance_simulation_context(interpolatedControl, trajectory_idx, controlCost, targetHoldIDs, false, false, nStates);

					float stateCost=0;
					if (physicsBroken)
					{
						stateCost = 1e20;
						disconnectHandsFeetJoints(trajectory_idx);
						restoreOdeState(masterContextID);
						connectHandsFeetJoints(trajectory_idx);
					}
					else
					{
						//compute state cost, only including the hold costs at the last step
						//bool includeTarget = false;
						//if (segmentIdx == lastSegment - 1)
						//{
						//	includeTarget = true;
						//}

//						if (segmentIdx >= lastSegment - 1)
	//					{
						std::vector<bool> mAllowedRelease;
						mAllowedRelease.reserve(4);
						for (int i=0; i<4; i++)
						{
							mAllowedRelease.push_back(cmaesResults[trajectory_idx].nSteps > nLetGoStep[trajectory_idx][i]);
						}

						mConnectDisconnectContactPoint(targetHoldIDs, trajectory_idx, 30, mAllowedRelease);
		//				}

						stateCost = computeStateCost(mContextController, sourcePos, targetPos, targetHoldIDs
							, false, trajectory_idx); // , includeTarget, segmentIdx >= lastSegment

					}

					// this is the cost of all trajectory
					cost += (stateCost + controlCost);

					int cStep = cmaesResults[trajectory_idx].nSteps - 1;
					saveBipedStateSimulationTrajectory(trajectory_idx, cStep);

				} //for each step in segment

				mContextController->resultPosition[trajectory_idx] = mContextController->getEndPointPosBones(SimulationContext::BodyName::BodyTrunk);

			};  //lambda for simulating sample

			if (useThreads)
				worker_queue.push_back(std::async(std::launch::async,simulate_sample,sampleIdx));
			else
				simulate_sample(sampleIdx);


		} //for each cample
		if (useThreads)
		{
			for (std::future<void>& is_ready : worker_queue)
			{
				is_ready.wait();
			}
		}

		//find min cost, convert costs to goodnesses through negating, and update cmaes
		float minCost = FLT_MAX;
		int bestIdx = 0;
		for (int sampleIdx=0; sampleIdx < nCurrentTrajectories; sampleIdx++)
		{
			cmaesResults[sampleIdx].cost=cmaesSamples[sampleIdx].second;
			if (cmaesSamples[sampleIdx].second < minCost && cmaesSamples[sampleIdx].second >= 0)
			{
				bestIdx=sampleIdx;
				minCost = cmaesSamples[sampleIdx].second;
			}
			cmaesSamples[sampleIdx].second*=-1.0f;
		}
		current_cost_state = minCost;

		//rcPrintString("Traj. cost for controller: %f", current_cost_state);
//		if (showDebugInfo) debug_visualize(0);

		//Remember if this iteration produced the new best one. This is needed just in case CMAES loses it in the next iteration.
		//For example, at the end of the optimization, a reaching hand might fluctuate in and out of the hold, and the results will be rejected
		//if the hand is not reaching the hold when iteration stopped
		if (firstIter || minCost < bestCmaesTrajectory.cost)
		{
			bestCmaesTrajectory = cmaesResults[bestIdx];
			setCurrentOdeContext(bestIdx);
			for (int step = 0; step < bestCmaesTrajectory.nSteps; step++)
			{
				restoreOdeState(states_trajectory_steps[bestIdx][step].saving_slot_state);
				saveOdeState(states_trajectory_steps[nTrajectories][step].saving_slot_state, bestIdx);
				states_trajectory_steps[nTrajectories][step].hold_bodies_ids = states_trajectory_steps[bestIdx][step].hold_bodies_ids;
				for (int i = 0; i < 4; i++)
				{
					states_trajectory_steps[nTrajectories][step].forces[i] = states_trajectory_steps[bestIdx][step].forces[i];
					counter_let_go[i][nTrajectories] = counter_let_go[i][bestIdx];
				}
			}
		}
		if (mContextController->getHoldBodyIDs(2, bestIdx) == -1 && mContextController->getHoldBodyIDs(3, bestIdx) == -1)
		{
			rcPrintString("!!!!!!!!!!!!!!!! let go of foot !!!!!!!!!!!!!!!!!!!!!");
		}
		//bestCmaEsTrajectoryIdx=bestIdx;
		cmaes.update(cmaesSamples,false);

		//visualize
		setCurrentOdeContext(bestIdx);
		float stateCost = computeStateCost(mContextController, sourcePos, targetPos, targetHoldIDs, showDebugInfo, bestIdx); // true
		rcPrintString("Traj. cost for controller: %f", stateCost);
		mContextController->mDrawStuff(-1,-1,bestIdx,true,false);
		if (showDebugInfo) debugVisulizeForceOnHnadsFeet(bestIdx);

		setCurrentOdeContext(masterContextID);
		return;
	}

	// sync all contexts with the beginning state of the optimization
	void syncMasterContextWithStartState(bool loadAnyWay = true)
	{
		int cOdeContext = getCurrentOdeContext();

		setCurrentOdeContext(ALLTHREADS);

		bool flag_set_state = disconnectHandsFeetJoints(ALLTHREADS);

		restoreOdeState(startState.saving_slot_state, false);
		saveOdeState(masterContextID,0);
		mContextController->saveContextIn(startState);
		startState.saving_slot_state = masterContextID;

		connectHandsFeetJoints(ALLTHREADS);

		setCurrentOdeContext(cOdeContext);

		return;
	}

	void reset()
	{
		return;
	}

private:

	bool isSetAEqualsSetB(const std::vector<int>& set_a, const std::vector<int>& set_b)
	{
		//AALTO_ASSERT1(set_a.size()==4 && set_b.size()==4);
		int diff=0;
		diff+=abs(set_a[0]-set_b[0]);
		diff+=abs(set_a[1]-set_b[1]);
		diff+=abs(set_a[2]-set_b[2]);
		diff+=abs(set_a[3]-set_b[3]);
		return diff==0;
	}

	void clampCMAESSegmentControls(int segmentIdx,bool clampDuration, int nVelocities,int nValuesPerSegment, VectorXf &sample)
	{
		int segmentStart=segmentIdx*nValuesPerSegment;
		if (clampDuration)
		{
			sample[segmentStart]=clipMinMaxf(sample[segmentStart],minSegmentDuration,maxSegmentDuration);  
			segmentStart++;
		}
		if (cmaesSamplePoses)
		{
			for (int velIdx=0; velIdx<nVelocities; velIdx++)
			{
				sample[segmentStart+velIdx]=clipMinMaxf(sample[segmentStart+velIdx],poseMin[velIdx],poseMax[velIdx]);
			}
		}
		else
		{
			for (int velIdx=0; velIdx<nVelocities; velIdx++)
			{
				sample[segmentStart+velIdx]=clipMinMaxf(sample[segmentStart+velIdx],controlMin[velIdx],controlMax[velIdx]);
			}
		}
		for (int fmaxIdx=0; fmaxIdx<fmCount; fmaxIdx++)
		{
			//safest to start with high fmax and let the optimizer decrease them later, thus the mean at maximumForce
			sample[segmentStart+nVelocities+fmaxIdx]=clipMinMaxf(sample[segmentStart+nVelocities+fmaxIdx],minimumForce,maximumForce);
		}
	}
	
	bool advance_simulation_context(Eigen::VectorXf& cControl, int trajectory_idx, float& controlCost, std::vector<int>& dHoldIDs,
		bool debugPrint, bool flagSaveSlots, std::vector<BipedState>& nStates)
	{
		bool physicsBroken = false;

		//apply control
		apply_control(mContextController, cControl);

		physicsBroken = !stepOde(timeStep,false);

		if (flagSaveSlots && !physicsBroken)
		{
			BipedState nState = this->startState.getNewCopy(mContextController->getNextFreeSavingSlot(), trajectory_idx);
			nStates.push_back(nState);
		}
		if (!physicsBroken)
		{
			controlCost += compute_control_cost(mContextController, trajectory_idx); // masterContextID
		}

		if (debugPrint) rcPrintString("Control cost for controller: %f",controlCost);

		if (!physicsBroken)
		{
			return true;
		}

		return false;
	} 

	Eigen::VectorXf getBestControl(int cTimeStep)
	{
		return bestCmaesTrajectory.control[cTimeStep];
	}

	int getNSteps()
	{
		return bestCmaesTrajectory.nSteps;
	}

	void apply_control(SimulationContext* iContextCPBP, const Eigen::VectorXf& control)
	{
		for (int j = 0; j < iContextCPBP->getJointSize(); j++)
		{
			float c_j = control[j];
			if (cmaesSamplePoses)
			{
				iContextCPBP->driveMotorToPose(j,c_j);
			}
			else
			{
				iContextCPBP->setMotorSpeed(j, c_j);
			}
		}
		if (fmCount!=0)
		{
			//fmax values for each group are after the joint motor speeds
			iContextCPBP->setMotorGroupFmaxes(&control[iContextCPBP->getJointSize()]);
		}
	}
	
};

class mSampleStructure
{
public:
	void initialization()
	{
		toNodeGraph = -1;
		fromNodeGraph = -1;

		cOptimizationCost = FLT_MAX;
		numItrFixedCost = 0;

		isOdeConstraintsViolated = false;
		isReached = false;
		isRejected = false;

		// variables for recovery from playing animation
		restartSampleStartState = false;

		/*for (int i = 0; i < 4 ; i++)
		{
			to_hold_ids.push_back(-1);
		}*/

		starting_time = 0;
		toNode = -1;

		control_cost = 0.0f;
	}

	mSampleStructure()
	{
		initialization();
		isSet = false;

		closest_node_index = -1;
	}

	mSampleStructure(std::vector<mOptCPBP::ControlledPoses>& iSourceP, std::vector<Vector3>& iDestinationP,
					 std::vector<int>& iInitialHoldIDs, std::vector<int>& iDesiredHoldIDs, int iClosestIndexNode,
					 std::vector<Vector3>& iWorkSpaceHolds, std::vector<Vector3>& iWorkSpaceColor, std::vector<Vector3>& iDPoint)
	{
		initialization();
		isSet = true;

		for (unsigned int i = 0; i < iSourceP.size(); i++)
		{
			sourceP.push_back(iSourceP[i]);
			destinationP.push_back(iDestinationP[i]);
		}

		initial_hold_ids = iInitialHoldIDs;
		desired_hold_ids = iDesiredHoldIDs;

		closest_node_index = iClosestIndexNode;

		// for debug visualization
		mWorkSpaceHolds = iWorkSpaceHolds;
		mWorkSpaceColor = iWorkSpaceColor;
		dPoint = iDPoint;
	}

	void draw_ws_points(Vector3& _from)
	{
		//for (unsigned int i = 0; i < mWorkSpaceHolds.size(); i++)
		//{
		//	Vector3 _to = mWorkSpaceHolds[i];
		//	rcSetColor(mWorkSpaceColor[i].x(), mWorkSpaceColor[i].y(), mWorkSpaceColor[i].z());
		//	SimulationContext::drawLine(_from, _to);
		//}

		Vector3 color(0,0,1);
		float mCubeSize = 0.1f;
		for (unsigned int i = 0; i < dPoint.size(); i++)
		{
			rcSetColor(color.x(), color.y(), color.z());
			SimulationContext::drawCube(dPoint[i], mCubeSize);
		}
		return;
	}

	void drawDesiredTorsoDir(Vector3& _from) // if the last element of source point is torsoDir we have a desired direction
	{
		if (sourceP.size() > 0)
		{
			if (sourceP[sourceP.size() - 1] == mOptCPBP::ControlledPoses::TorsoDir)
			{
				rcSetColor(1.0f,0.0f,0.0f);
				Vector3 _to = _from + destinationP[destinationP.size() - 1];
				SimulationContext::drawLine(_from, _to);
			}
		}
	}

	std::vector<mOptCPBP::ControlledPoses> sourceP; // contains head, trunk and contact points sources
	std::vector<Vector3> destinationP; // contains head's desired angle, and trunk's and contact points's desired positions (contact points's desired positions have the most value to us)

	std::vector<int> desired_hold_ids; // desired holds's ids to reach
	std::vector<int> initial_hold_ids; // connected joints to (ll,rl,lh,rh); -1 means it is disconnected, otherwise it is connected to the hold with the same id

	float starting_time;
	//std::vector<int> to_hold_ids; // just for debugging
	int toNode; // from closest node to toNode

	int toNodeGraph;
	int fromNodeGraph;

	// check cost change
	float cOptimizationCost;
	int numItrFixedCost;

	int closest_node_index;
	std::vector<BipedState> statesFromTo;
	
	bool isOdeConstraintsViolated;
	bool isReached;
	bool isRejected;
	bool isSet;

	// for debug visualization
	std::vector<Vector3> mWorkSpaceHolds;
	std::vector<Vector3> mWorkSpaceColor;
	std::vector<Vector3> dPoint;

	// variables for recovery from playing animation
	bool restartSampleStartState;

	// handling energy cost - control cost
	float control_cost;
};

class mNode
{
private:
	float costFromFatherToThis(std::vector<BipedState>& _fromFatherToThis)
	{
		float _cost = 0;
		for (int i = 0; i < (int)(_fromFatherToThis.size() - 1); i++)
		{
			_cost += getDisFrom(_fromFatherToThis[i], _fromFatherToThis[i+1]);
		}
		return _cost;
	}

	float getDisFrom(BipedState& c1, BipedState& c2) // used for calculating cost
	{
		float dis = 0.0f;
		for (unsigned int i = 0; i < c1.bodyStates.size(); i++)
		{
			BodyState m_b_i = c1.bodyStates[i];
			BodyState c_b_i = c2.bodyStates[i];
			dis += (m_b_i.getPos() - c_b_i.getPos()).squaredNorm();
		}
		return sqrtf(dis);
	}
public:
	mNode(BipedState& iState, int iFatherIndex, int iNodeIndex, std::vector<BipedState>& _fromFatherToThis)
	{
		cNodeInfo = iState;

		statesFromFatherToThis = _fromFatherToThis;

		mFatherIndex = iFatherIndex;
		nodeIndex = iNodeIndex;

		cCost = 0.0f;
		control_cost = 0.0f;
	}

	bool isNodeEqualTo(std::vector<int>& s_i)
	{
		if (mNode::isSetAEqualsSetB(s_i, cNodeInfo.hold_bodies_ids))
		{
			return true;
		}
		if (mNode::isSetAEqualsSetB(s_i, poss_hold_ids))
		{
			return true;
		}
		return false;
	}

	// distance to the whole body state
	float getDisFrom(BipedState& c) // used for adding node to the tree
	{
		float dis = 0.0f;
		for (unsigned int i = 0; i < cNodeInfo.bodyStates.size(); i++)
		{
			BodyState m_b_i = cNodeInfo.bodyStates[i];
			BodyState c_b_i = c.bodyStates[i];
			dis += (m_b_i.getPos() - c_b_i.getPos()).squaredNorm();
			//dis += (m_b_i.vel - c_b_i.vel).LengthSquared();
			//dis += squared(m_b_i.angle - c_b_i.angle);
			//dis += squared(m_b_i.aVel - c_b_i.aVel);
		}
		return sqrtf(dis);
	}

	// just given middle trunk pose or hand pos
	float getSumDisEndPosTo(Vector3& iP)
	{
		float dis = 0.0f;
		dis += (cNodeInfo.getEndPointPosBones(SimulationContext::ContactPoints::LeftLeg) - iP).squaredNorm();
		dis += (cNodeInfo.getEndPointPosBones(SimulationContext::ContactPoints::RightLeg)- iP).squaredNorm();
		dis += (cNodeInfo.getEndPointPosBones(SimulationContext::ContactPoints::LeftArm)- iP).squaredNorm();
		dis += (cNodeInfo.getEndPointPosBones(SimulationContext::ContactPoints::RightArm)- iP).squaredNorm();
		dis += (cNodeInfo.getEndPointPosBones(SimulationContext::BodyName::BodyTrunk)- iP).squaredNorm();
		return sqrtf(dis);
	}

	float getCostDisconnectedArmsLegs()
	{
		int counter_disconnected_legs = 0;
		int counter_disconnected_arms = 0;
		for (unsigned int i = 0; i < cNodeInfo.hold_bodies_ids.size(); i++)
		{
			if (i <= 1 && cNodeInfo.hold_bodies_ids[i] == -1)
				counter_disconnected_legs++;
			if (i > 1 && cNodeInfo.hold_bodies_ids[i] == -1)
				counter_disconnected_arms++;
		}
		return 10.0f * (counter_disconnected_legs + counter_disconnected_arms);
	}

	float getCostNumMovedLimbs(std::vector<int>& iStance)
	{
		int mCount = getDiffBtwSetASetB(iStance, this->cNodeInfo.hold_bodies_ids);
		
		return 1.0f * mCount;
	}

	float getSumDisEndPosTo(std::vector<int>& iIDs, std::vector<Vector3>& iP)
	{
		float dis = 0.0f;
		
		for (unsigned int i = 0; i < iIDs.size(); i++)
		{
			if (iIDs[i] != -1)
			{
				dis += (cNodeInfo.getEndPointPosBones(SimulationContext::ContactPoints::LeftLeg + i) - iP[i]).squaredNorm();
			}
		}
		
		return dis;
	}
	
	bool addTriedHoldSet(std::vector<int>& iSample_desired_hold_ids)
	{
		if (!isInTriedHoldSet(iSample_desired_hold_ids))
		{
			std::vector<int> nSample;
			for (unsigned int i = 0; i < iSample_desired_hold_ids.size(); i++)
			{
				nSample.push_back(iSample_desired_hold_ids[i]);
			}
			mTriedHoldSet.push_back(nSample);
			return true;
		}
		return false;
	}

	bool isInTriedHoldSet(std::vector<int>& iSample_desired_hold_ids)
	{
		for (unsigned int i = 0; i < mTriedHoldSet.size(); i++)
		{
			std::vector<int> t_i = mTriedHoldSet[i];
			bool flag_found_try = isSetAEqualsSetB(t_i, iSample_desired_hold_ids);
			if (flag_found_try)
			{
				return true;
			}
		}
		return false;
	}

	static bool isSetAGreaterThan(std::vector<int>& set_a, int f)
	{
		for (unsigned int j = 0; j < set_a.size(); j++)
		{
			if (set_a[j] <= f)
			{
				return false;
			}
		}
		return true;
	}

	static bool isSetAEqualsSetB(const std::vector<int>& set_a, const std::vector<int>& set_b)
	{
		//AALTO_ASSERT1(set_a.size()==4 && set_b.size()==4);
		if (set_a.size()==4 && set_b.size()==4)
		{
			int diff=0;
			diff+=abs(set_a[0]-set_b[0]);
			diff+=abs(set_a[1]-set_b[1]);
			diff+=abs(set_a[2]-set_b[2]);
			diff+=abs(set_a[3]-set_b[3]);
			return diff==0;
		}
		for (unsigned int i = 0; i < set_a.size(); i++)
		{
			if (set_a[i] != set_b[i])
			{
				return false;
			}
		}

		return true;

	}

	static int getDiffBtwSetASetB(std::vector<int>& set_a, std::vector<int>& set_b)
	{
		int mCount = 0;
		for (unsigned int i = 0; i < set_a.size(); i++)
		{
			if (set_a[i] != set_b[i])
			{
				mCount++;
			}
		}
		return mCount;
	}

	BipedState cNodeInfo;
	std::vector<BipedState> statesFromFatherToThis;

	std::vector<int> poss_hold_ids;
	// variables for building the tree
	int nodeIndex;
	int mFatherIndex; // -1 means no father or root node
	std::vector<int> mChildrenIndices; // vector is empty

	std::vector<std::vector<int>> mTriedHoldSet;

	float cCost;
	float control_cost; // handling energy consumption
};

#include "mKDTree.h"
class mSampler
{
public:
	float climberRadius;
	float climberLegLegDis;
	float climberHandHandDis;

	KDTree<int> mHoldsKDTree;
	std::vector<Vector3> myHoldPoints;
	
	// helping sampling
	std::vector<std::vector<int>> indices_higher_than;
	std::vector<std::vector<int>> indices_lower_than;

	mSampler(SimulationContext* iContextRRT)
		:mHoldsKDTree(3)
	{
		climberRadius = iContextRRT->getClimberRadius();
		climberLegLegDis = iContextRRT->getClimberLegLegDis();
		climberHandHandDis = iContextRRT->getClimberHandHandDis();

		for (unsigned int i = 0; i < iContextRRT->holds_body.size(); i++)
		{
			Vector3 hPos = iContextRRT->getHoldPos(i);

			myHoldPoints.push_back(hPos);
			mHoldsKDTree.insert(getHoldKey(hPos), myHoldPoints.size() - 1);
		}

		fillInLowerHigherHoldIndices();
	}

	/////////////////////////////////////////////////////// sample costs ////////////////////////////////////////////////////////////////////
	float getSampleCost(std::vector<int>& desiredStance, std::vector<int>& sampledPriorStance)
	{
		float iDis = 0.0f;
		for (unsigned int j = 0; j < desiredStance.size(); j++)
		{
			if (sampledPriorStance[j] != -1)
			{
				iDis += (getHoldPos(desiredStance[j]) - getHoldPos(sampledPriorStance[j])).norm();
			}
			else
			{
				iDis += 10.0f;
			}
		}
		return iDis;
	}

	bool isFromStanceToStanceValid(std::vector<int>& _formStanceIds, std::vector<int>& _toStanceIds, bool isInitialStance)
	{
		if (!isAllowedHandsLegsInDSample(_formStanceIds, _toStanceIds, isInitialStance)) // are letting go of hands and legs allowed
		{
			return false;	
		}

		std::vector<Vector3> sample_n_hold_points; float size_n = 0;
		Vector3 midPointN = getHoldStancePosFrom(_toStanceIds, sample_n_hold_points, size_n);
		if (size_n == 0)
		{
			return false;
		}

		if (!acceptDirectionLegsAndHands(midPointN, _toStanceIds, sample_n_hold_points))
		{
			return false;
		}

		if (!isFromStanceCloseEnough(_formStanceIds, _toStanceIds))
		{
			return false;
		}

		if (!earlyAcceptOfSample(_toStanceIds, isInitialStance)) // is it kinematically reachable
		{
			return false;
		}
		return true;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	Vector3 getHoldPos(int i)
	{
		if (myHoldPoints.size() == 0)
		{
			return Vector3(0.0f, 0.0f, 0.0f);
		}
		Vector3 rPoint = myHoldPoints[0];
		if (i >= 0)
		{
			rPoint = myHoldPoints[i];
		}
		return rPoint;
	}

	Vector3 getHoldStancePosFrom(std::vector<int>& sample_desired_hold_ids, std::vector<Vector3>& sample_desired_hold_points, float& mSize)
	{
		Vector3 midPoint(0.0f,0.0f,0.0f);
		for (unsigned int i = 0; i < sample_desired_hold_ids.size(); i++)
		{
			if (sample_desired_hold_ids[i] != -1)
			{
				sample_desired_hold_points.push_back(getHoldPos(sample_desired_hold_ids[i]));
				midPoint += sample_desired_hold_points[i];
				mSize++;
			}
			else
				sample_desired_hold_points.push_back(Vector3(0,0,0));
		}

		if (mSize > 0)
		{
			midPoint = midPoint / mSize;
		}

		return midPoint;
	}

	std::vector<std::vector<int>> getListOfSamples(mNode* closestNode, std::vector<int>& iSample_desired_hold_ids, bool isInitialStance)
	{
		std::vector<std::vector<int>> list_samples;

		std::vector<Vector3> sample_Des_hold_points; float size_d = 0;
		Vector3 midPointD = getHoldStancePosFrom(iSample_desired_hold_ids, sample_Des_hold_points, size_d);

		std::vector<std::vector<size_t>> workSpaceHoldIDs;
		std::vector<Vector3> workspacePoints; // for debug visualization
		std::vector<Vector3> workSpacePointColor; // for debug visualization
		std::vector<bool> isIDInWorkSapceVector; // is current id in workspace or not

		getAllHoldsInRangeAroundAgent(closestNode, workSpaceHoldIDs, workspacePoints, workSpacePointColor, isIDInWorkSapceVector);

		std::vector<int> initial_holds_ids = closestNode->cNodeInfo.hold_bodies_ids;

		std::vector<int> diff_hold_index;
		std::vector<int> same_hold_index;

		for (unsigned int i = 0; i < iSample_desired_hold_ids.size(); i++)
		{
			if (initial_holds_ids[i] != iSample_desired_hold_ids[i] || initial_holds_ids[i] == -1)
			{
				diff_hold_index.push_back(i);
			}
			else
			{
				same_hold_index.push_back(i);
			}
		}

		std::vector<std::vector<int>> possible_hold_index_diff;
		std::vector<unsigned int> itr_index_diff;
		for (unsigned int i = 0; i < diff_hold_index.size(); i++)
		{
			std::vector<int> possible_hold_diff_i;
			int index_diff_i = diff_hold_index[i];

			addToSetHoldIDs(-1, possible_hold_diff_i);
			addToSetHoldIDs(iSample_desired_hold_ids[index_diff_i], possible_hold_diff_i);
			
			if (!isInitialStance)
			{
				Vector3 cPos_i = closestNode->cNodeInfo.getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg + index_diff_i);
				int nHold_id = getClosest_WSHoldIndex_ToDesiredPos(workSpaceHoldIDs[index_diff_i], iSample_desired_hold_ids[index_diff_i], initial_holds_ids[index_diff_i], cPos_i);
				addToSetHoldIDs(nHold_id, possible_hold_diff_i);
			}

			addToSetHoldIDs(initial_holds_ids[index_diff_i], possible_hold_diff_i);

			itr_index_diff.push_back(0);
			possible_hold_index_diff.push_back(possible_hold_diff_i);
		}

		bool flag_continue = true;

		if (diff_hold_index.size() == 0)
		{
			flag_continue = false;
			list_samples.push_back(iSample_desired_hold_ids);
		}

		while (flag_continue)
		{
			// create sample n
			std::vector<int> sample_n;
			for (int i = iSample_desired_hold_ids.size() - 1; i >= 0; i--)
			{
				sample_n.push_back(-1);
			}
			for (unsigned int i = 0; i < diff_hold_index.size(); i++)
			{
				int index_diff_i = diff_hold_index[i];
				int itr_index_diff_i = itr_index_diff[i];
				std::vector<int> possible_hold_diff_i = possible_hold_index_diff[i];
				sample_n[index_diff_i] = possible_hold_diff_i[itr_index_diff_i];
			}
			for (unsigned int i = 0; i < same_hold_index.size(); i++)
			{
				int index_same_i = same_hold_index[i];
				sample_n[index_same_i] = initial_holds_ids[index_same_i];
			}

			// increase itr num
			for (unsigned int i = 0; i < diff_hold_index.size(); i++)
			{
				itr_index_diff[i] = itr_index_diff[i] + 1;
				if (itr_index_diff[i] >= possible_hold_index_diff[i].size())
				{
					if (i == diff_hold_index.size() - 1)
					{
						flag_continue = false;
					}
					itr_index_diff[i] = 0;
				}
				else
				{
					break;
				}
			}

			///////////////////////////////////////////////////////////////
			// prior for adding sample_n to the list of possible samples //
			///////////////////////////////////////////////////////////////
			if (!isAllowedHandsLegsInDSample(initial_holds_ids, sample_n, isInitialStance)) // are letting go of hands and legs allowed
			{
				continue;	
			}

			std::vector<Vector3> sample_n_hold_points; float size_n = 0;
			Vector3 midPointN = getHoldStancePosFrom(sample_n, sample_n_hold_points, size_n);
			if (size_n == 0)
			{
				continue;
			}

			if (!acceptDirectionLegsAndHands(midPointN, sample_n, sample_n_hold_points))
			{
				continue;
			}

			if (!isFromStanceCloseEnough(initial_holds_ids, sample_n))
			{
				continue;
			}

			if (!earlyAcceptOfSample(sample_n, isInitialStance)) // is it kinematically reachable
			{
				continue;
			}
			if (closestNode->isInTriedHoldSet(sample_n)) // is it tried
			{
				continue;
			}

			list_samples.push_back(sample_n);
		}

		return list_samples;
	}

	std::vector<std::vector<int>> getListOfStanceSamples(std::vector<int>& from_hold_ids, std::vector<int>& to_hold_ids, bool isInitialStance)
	{
		std::vector<std::vector<int>> list_samples;

		std::vector<int> initial_holds_ids = from_hold_ids;

		std::vector<int> diff_hold_index;
		std::vector<int> same_hold_index;

		for (unsigned int i = 0; i < to_hold_ids.size(); i++)
		{
			if (initial_holds_ids[i] != to_hold_ids[i] || initial_holds_ids[i] == -1)
			{
				diff_hold_index.push_back(i);
			}
			else
			{
				same_hold_index.push_back(i);
			}
		}

		std::vector<std::vector<int>> possible_hold_index_diff;
		std::vector<unsigned int> itr_index_diff;
		for (unsigned int i = 0; i < diff_hold_index.size(); i++)
		{
			std::vector<int> possible_hold_diff_i;
			int index_diff_i = diff_hold_index[i];

			addToSetHoldIDs(-1, possible_hold_diff_i);
			addToSetHoldIDs(to_hold_ids[index_diff_i], possible_hold_diff_i);

			addToSetHoldIDs(initial_holds_ids[index_diff_i], possible_hold_diff_i);

			itr_index_diff.push_back(0);
			possible_hold_index_diff.push_back(possible_hold_diff_i);
		}

		bool flag_continue = true;

		if (diff_hold_index.size() == 0)
		{
			flag_continue = false;
			list_samples.push_back(to_hold_ids);
		}

		while (flag_continue)
		{
			// create sample n
			std::vector<int> sample_n;
			for (int i = to_hold_ids.size() - 1; i >= 0; i--)
			{
				sample_n.push_back(-1);
			}
			for (unsigned int i = 0; i < diff_hold_index.size(); i++)
			{
				int index_diff_i = diff_hold_index[i];
				int itr_index_diff_i = itr_index_diff[i];
				std::vector<int> possible_hold_diff_i = possible_hold_index_diff[i];
				sample_n[index_diff_i] = possible_hold_diff_i[itr_index_diff_i];
			}
			for (unsigned int i = 0; i < same_hold_index.size(); i++)
			{
				int index_same_i = same_hold_index[i];
				sample_n[index_same_i] = initial_holds_ids[index_same_i];
			}

			// increase itr num
			for (unsigned int i = 0; i < diff_hold_index.size(); i++)
			{
				itr_index_diff[i] = itr_index_diff[i] + 1;
				if (itr_index_diff[i] >= possible_hold_index_diff[i].size())
				{
					if (i == diff_hold_index.size() - 1)
					{
						flag_continue = false;
					}
					itr_index_diff[i] = 0;
				}
				else
				{
					break;
				}
			}

			///////////////////////////////////////////////////////////////
			// prior for adding sample_n to the list of possible samples //
			///////////////////////////////////////////////////////////////
			if (!isAllowedHandsLegsInDSample(initial_holds_ids, sample_n, isInitialStance)) // are letting go of hands and legs allowed
			{
				continue;	
			}

			std::vector<Vector3> sample_n_hold_points; float size_n = 0;
			Vector3 midPointN = getHoldStancePosFrom(sample_n, sample_n_hold_points, size_n);
			if (size_n == 0)
			{
				continue;
			}

			if (!acceptDirectionLegsAndHands(midPointN, sample_n, sample_n_hold_points))
			{
				continue;
			}

			if (!isFromStanceCloseEnough(initial_holds_ids, sample_n))
			{
				continue;
			}

			if (!earlyAcceptOfSample(sample_n, isInitialStance)) // is it kinematically reachable
			{
				continue;
			}

			list_samples.push_back(sample_n);
		}

		return list_samples;
	}

	void getListOfStanceSamplesAround(int to_hold_id, std::vector<int>& from_hold_ids, std::vector<std::vector<int>> &out_list_samples)
	{
		out_list_samples.clear();
		static std::vector<std::vector<int>> possible_hold_index_diff;
		if (possible_hold_index_diff.size()!=4)
			possible_hold_index_diff.resize(4);
		const int itr_index_diff_size=4;
		int itr_index_diff[itr_index_diff_size]={0,0,0,0};

		for (unsigned int i = 0; i < 4; i++)
		{
			std::vector<int> &possible_hold_diff_i=possible_hold_index_diff[i];
			possible_hold_diff_i.clear();

			addToSetHoldIDs(-1, possible_hold_diff_i);
			addToSetHoldIDs(from_hold_ids[i], possible_hold_diff_i);

			for (unsigned int j = 0; j < indices_lower_than[to_hold_id].size(); j++)
			{
				addToSetHoldIDs(indices_lower_than[to_hold_id][j], possible_hold_diff_i);
			}

			for (unsigned int j = 0; j < from_hold_ids.size(); j++)
			{
				if (mSampler::isInSetHoldIDs(from_hold_ids[j], indices_lower_than[to_hold_id]))
					addToSetHoldIDs(from_hold_ids[j], possible_hold_diff_i);
			}
		}
		bool flag_continue = true;
		while (flag_continue)
		{
			// create sample n
			std::vector<int> sample_n(4,-1);

			for (unsigned int i = 0; i < itr_index_diff_size; i++)
			{
				int itr_index_diff_i = itr_index_diff[i];
				std::vector<int> &possible_hold_diff_i = possible_hold_index_diff[i];
				sample_n[i] = possible_hold_diff_i[itr_index_diff_i];
			}

			// increase itr num
			for (unsigned int i = 0; i < itr_index_diff_size; i++)
			{
				itr_index_diff[i] = itr_index_diff[i] + 1;
				if (itr_index_diff[i] >= (int)possible_hold_index_diff[i].size())
				{
					if (i == itr_index_diff_size - 1)
					{
						flag_continue = false;
					}
					itr_index_diff[i] = 0;
				}
				else
				{
					break;
				}
			}

			///////////////////////////////////////////////////////////////
			// prior for adding sample_n to the list of possible samples //
			///////////////////////////////////////////////////////////////

			if (sample_n[2] != to_hold_id && sample_n[3] != to_hold_id)
			{
				continue;
			}

			if (!isFromStanceToStanceValid(from_hold_ids, sample_n, false))
			{
				continue;
			}

			if (!isInSampledStanceSet(sample_n, out_list_samples))
				out_list_samples.push_back(sample_n);
		}
	}

	//this version not used anymore?
	std::vector<std::vector<int>> getListOfStanceSamplesAround(std::vector<int>& from_hold_ids, bool isInitialStance)
	{
		std::vector<std::vector<int>> list_samples;

		// this function assumes all from_hold_ids are greater than -1
		std::vector<std::vector<int>> workSpaceHoldIDs;
		getAllHoldsInRangeAroundAgent(from_hold_ids, workSpaceHoldIDs);

		std::vector<std::vector<int>> possible_hold_index_diff;
		std::vector<unsigned int> itr_index_diff;
		for (unsigned int i = 0; i < workSpaceHoldIDs.size(); i++)
		{
			std::vector<int> possible_hold_diff_i;

			std::vector<int> w_i = workSpaceHoldIDs[i];

			for (unsigned int j = 0; j < w_i.size(); j++)
			{
				addToSetHoldIDs(w_i[j], possible_hold_diff_i);
			}

			addToSetHoldIDs(from_hold_ids[i], possible_hold_diff_i);

			itr_index_diff.push_back(0);
			possible_hold_index_diff.push_back(possible_hold_diff_i);
		}

		bool flag_continue = true;
		while (flag_continue)
		{
			// create sample n
			std::vector<int> sample_n;
			for (int i = from_hold_ids.size() - 1; i >= 0; i--)
			{
				sample_n.push_back(-1);
			}
			for (unsigned int i = 0; i < itr_index_diff.size(); i++)
			{
				int itr_index_diff_i = itr_index_diff[i];
				std::vector<int> possible_hold_diff_i = possible_hold_index_diff[i];
				sample_n[i] = possible_hold_diff_i[itr_index_diff_i];
			}

			// increase itr num
			for (unsigned int i = 0; i < itr_index_diff.size(); i++)
			{
				itr_index_diff[i] = itr_index_diff[i] + 1;
				if (itr_index_diff[i] >= possible_hold_index_diff[i].size())
				{
					if (i == itr_index_diff.size() - 1)
					{
						flag_continue = false;
					}
					itr_index_diff[i] = 0;
				}
				else
				{
					break;
				}
			}

			///////////////////////////////////////////////////////////////
			// prior for adding sample_n to the list of possible samples //
			///////////////////////////////////////////////////////////////

			if (!isFromStanceToStanceValid(from_hold_ids, sample_n, isInitialStance))
			{
				continue;
			}

			list_samples.push_back(sample_n);
		}
		return list_samples;
	}

	mSampleStructure getSampleFrom(mNode* closestNode, std::vector<int>& chosen_desired_HoldsIds, bool flag_forShowing)
	{
		if (closestNode == nullptr)
		{
			return mSampleStructure();
		}

		if (flag_forShowing)
		{
			mSampleStructure ret_sample;
			ret_sample.closest_node_index = closestNode->nodeIndex;
			for (unsigned int i = 0; i < chosen_desired_HoldsIds.size(); i++)
			{
				if (chosen_desired_HoldsIds[i] >= 0)
					ret_sample.dPoint.push_back(getHoldPos(chosen_desired_HoldsIds[i]));
			}
			return ret_sample;
		}

		std::vector<std::vector<size_t>> workSpaceHoldIDs;
		std::vector<Vector3> workspacePoints; // for debug visualization
		std::vector<Vector3> workSpacePointColor; // for debug visualization
		std::vector<bool> isIDInWorkSapceVector; // is current id in workspace or not

		getAllHoldsInRangeAroundAgent(closestNode, workSpaceHoldIDs, workspacePoints, workSpacePointColor, isIDInWorkSapceVector);
		
		std::vector<int> rand_desired_HoldsIds;
		std::vector<int> sampled_rInitial_hold_ids;
		std::vector<Vector3> rndDesPos;
		Vector3 middle_point(0.0f, 0.0f, 0.0f);
		Vector3 max_point(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		float num_initiated_holds = 0;

		std::vector<int> initial_holds_ids = closestNode->cNodeInfo.hold_bodies_ids;
		std::vector<Vector3> dVisualPoints; // for debug visualization

		for (unsigned int i = 0; i < initial_holds_ids.size(); i++)
		{
			rand_desired_HoldsIds.push_back(chosen_desired_HoldsIds[i]);
			if (rand_desired_HoldsIds[i] == initial_holds_ids[i])
			{
				sampled_rInitial_hold_ids.push_back(initial_holds_ids[i]);
				if (initial_holds_ids[i] != -1)
				{
					rndDesPos.push_back(getHoldPos(initial_holds_ids[i]));
					middle_point += rndDesPos[i];
					if (max_point.z() < rndDesPos[i].z()){max_point.z() = rndDesPos[i].z();}
					num_initiated_holds++;
				}
				else
				{
					rndDesPos.push_back(Vector3(0.0f,0.0f,0.0f)); // some dummy var
				}
			}
			else
			{
				sampled_rInitial_hold_ids.push_back(-1);
				if (rand_desired_HoldsIds[i] != -1)
				{
					rndDesPos.push_back(getHoldPos(rand_desired_HoldsIds[i]));
					middle_point += rndDesPos[i];
					if (max_point.z() < rndDesPos[i].z()){max_point.z() = rndDesPos[i].z();}
					num_initiated_holds++;
				}
				else
				{
					rndDesPos.push_back(Vector3(0.0f,0.0f,0.0f)); // some dummy var
				}
			}
		}

		middle_point /= (num_initiated_holds + 0.001f);

		Vector3 desired_trunk_pos(middle_point.x(), middle_point.y(), max_point.z() - boneLength / 4.0f);

		rndDesPos.insert(rndDesPos.begin(), desired_trunk_pos);

		for (int i = rand_desired_HoldsIds.size() - 1; i >= 0; i--)
		{
			if (rand_desired_HoldsIds[i] != -1)
				dVisualPoints.push_back(getHoldPos(rand_desired_HoldsIds[i]));
		}

		closestNode->addTriedHoldSet(rand_desired_HoldsIds);

		std::vector<mOptCPBP::ControlledPoses> sourceP;
		std::vector<Vector3> destinationP;
		setDesiredPosForOptimization(sourceP, destinationP, rndDesPos, rand_desired_HoldsIds, true);

		return mSampleStructure(sourceP, destinationP, sampled_rInitial_hold_ids, rand_desired_HoldsIds, closestNode->nodeIndex, workspacePoints, workSpacePointColor, dVisualPoints);
	}

	static float getRandomBetween_01()
	{
		return ((float)rand()) / (float)RAND_MAX;
	}
	
	static int getRandomIndex(unsigned int iArraySize)
	{
		if (iArraySize == 0)
			return -1;
		int m_index = rand() % iArraySize;

		return m_index;
	}

	static bool isInSetHoldIDs(int hold_id, std::vector<int>& iSetIDs)
	{
		for (unsigned int i = 0; i < iSetIDs.size(); i++)
		{
			if (iSetIDs[i] == hold_id)
			{
				return true;
			}
		}
		return false;
	}
	
	static bool addToSetHoldIDs(int hold_id, std::vector<int>& iSetIDs)
	{
		if (!isInSetHoldIDs(hold_id, iSetIDs))
		{
			iSetIDs.push_back(hold_id);
			return true;
		}
		return false;
	}

	static void removeFromSetHoldIDs(int hold_id, std::vector<int>& iSetIDs)
	{
		for (unsigned int i = 0; i < iSetIDs.size(); i++)
		{
			if (iSetIDs[i] == hold_id)
			{
				iSetIDs.erase(iSetIDs.begin() + i);
				return;
			}
		}
	}

	static bool isInSampledStanceSet(std::vector<int>& sample_i, std::vector<std::vector<int>>& nStances)
	{
		for (unsigned int i = 0; i < nStances.size(); i++)
		{
			std::vector<int> t_i = nStances[i];
			bool flag_found_try = mNode::isSetAEqualsSetB(t_i, sample_i);
			if (flag_found_try)
			{
				return true;
			}
		}
		return false;
	}

private:

	void fillInLowerHigherHoldIndices()
	{
		float max_radius_around_hand = 1.0f * climberRadius;

		for (unsigned int  k = 0; k < myHoldPoints.size(); k++)
		{
			Vector3 dHoldPos = myHoldPoints[k];
			std::vector<size_t> ret_holds_ids = getPointsInRadius(myHoldPoints[k], max_radius_around_hand);
			std::vector<int> lower_holds_ids;
			std::vector<int> higher_holds_ids;
			for (unsigned int i = 0; i < ret_holds_ids.size(); i++)
			{
				bool flag_add = true;
				Vector3 hold_pos_i = getHoldPos(ret_holds_ids[i]);

				float cDis = (hold_pos_i - dHoldPos).norm();

				if (cDis < 0.01f)
				{
					lower_holds_ids.push_back(ret_holds_ids[i]);
					higher_holds_ids.push_back(ret_holds_ids[i]);
					continue;
				}

				float angle_btw_l = SimulationContext::getAngleBtwVectorsXZ(Vector3(1.0f, 0.0f, 0.0f), hold_pos_i - dHoldPos);
				if (((angle_btw_l <= 0) && (angle_btw_l >= -PI)) || (angle_btw_l >= PI)) 
				{
					lower_holds_ids.push_back(ret_holds_ids[i]);
				}
				if (((angle_btw_l >= 0 ) && (angle_btw_l <= PI)) || (angle_btw_l <= -PI)) 
				{
					higher_holds_ids.push_back(ret_holds_ids[i]);
				}
			}
			indices_lower_than.push_back(lower_holds_ids);
			indices_higher_than.push_back(higher_holds_ids);
		}
		return;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	int getClosest_WSHoldIndex_ToDesiredPos(std::vector<size_t>& workSpaceSet, int desiredID, int currentID, Vector3& cPos)
	{
		int ret_index = currentID;
		if (desiredID == -1)
			return ret_index;

		Vector3 dPos = getHoldPos(desiredID);
		float cDis = (dPos - cPos).norm();
		ret_index = currentID;
		for (unsigned int i = 0; i < workSpaceSet.size(); i++)
		{
			Vector3 nPosHold = getHoldPos(workSpaceSet[i]);
			float nDis = (dPos - nPosHold).norm();
			if (nDis < cDis)
			{
				ret_index = workSpaceSet[i];
				cDis = nDis;
			}
		}
		return ret_index;
	}

	bool earlyAcceptOfSample(std::vector<int>& sample_desired_hold_ids, bool isStanceGiven)
	{
		if (!isStanceGiven)
		{
			if (sample_desired_hold_ids[0] == sample_desired_hold_ids[1] && 
				(sample_desired_hold_ids[2] == sample_desired_hold_ids[0] || sample_desired_hold_ids[3] == sample_desired_hold_ids[0])) // if hands and legs are on the same hold
				return false;
			if (sample_desired_hold_ids[2] == sample_desired_hold_ids[3] && 
				(sample_desired_hold_ids[2] == sample_desired_hold_ids[0] || sample_desired_hold_ids[2] == sample_desired_hold_ids[1])) // if hands and legs are on the same hold
				return false;

			if (mNode::isSetAGreaterThan(sample_desired_hold_ids, -1))
			{
				std::vector<int> diff_hold_ids;
				for (unsigned int i = 0; i < sample_desired_hold_ids.size(); i++)
				{
					if (sample_desired_hold_ids[i] != -1)
						mSampler::addToSetHoldIDs(sample_desired_hold_ids[i], diff_hold_ids);
				}
				if (diff_hold_ids.size() <= 2)
				{
					return false;
				}
			}
		}
		std::vector<Vector3> sample_desired_hold_points;
		float mSize = 0;
		Vector3 midPoint = getHoldStancePosFrom(sample_desired_hold_ids, sample_desired_hold_points, mSize);

		if (mSize == 0)
		{
			return false;
		}

		// early reject of the sample (do not try the sample, because it is not reasonable)

		float coefficient_hand = 1.5f;
		float coefficient_leg = 1.2f;
		float coefficient_all = 1.1f;
		// if hands or legs distance are violating
		if (sample_desired_hold_ids[0] != -1 && sample_desired_hold_ids[1] != -1)
		{
			float dis_ll = (sample_desired_hold_points[0] - sample_desired_hold_points[1]).norm();
			if (dis_ll > coefficient_leg * climberLegLegDis)
			{
				return false;
			}
		}
		
		if (sample_desired_hold_ids[2] != -1 && sample_desired_hold_ids[3] != -1)
		{
			float dis_hh = (sample_desired_hold_points[2] - sample_desired_hold_points[3]).norm();
			if (dis_hh > coefficient_hand * climberHandHandDis)
			{
				return false;
			}
		}

		if (sample_desired_hold_ids[0] != -1 && sample_desired_hold_ids[2] != -1)
		{
			float dis_h1l1 = (sample_desired_hold_points[2] - sample_desired_hold_points[0]).norm();

			if (dis_h1l1 > coefficient_all * climberRadius)
				return false;
		}
		
		if (sample_desired_hold_ids[0] != -1 && sample_desired_hold_ids[3] != -1)
		{
			float dis_h2l1 = (sample_desired_hold_points[3] - sample_desired_hold_points[0]).norm();
			if (dis_h2l1 > coefficient_all * climberRadius)
				return false;
		}
		
		if (sample_desired_hold_ids[1] != -1 && sample_desired_hold_ids[2] != -1)
		{
			float dis_h1l2 = (sample_desired_hold_points[2] - sample_desired_hold_points[1]).norm();
			if (dis_h1l2 > coefficient_all * climberRadius)
				return false;
		}

		if (sample_desired_hold_ids[1] != -1 && sample_desired_hold_ids[3] != -1)
		{
			float dis_h2l2 = (sample_desired_hold_points[3] - sample_desired_hold_points[1]).norm();
			if (dis_h2l2 > coefficient_all * climberRadius)
				return false;
		}
		
		return true;
	}

	bool isFromStanceCloseEnough(std::vector<int>& initial_holds_ids, std::vector<int>& iSample_desired_hold_ids)
	{
		int m_count = 0; 
		for (unsigned int i = 0; i < iSample_desired_hold_ids.size(); i++)
		{
			if (iSample_desired_hold_ids[i] == initial_holds_ids[i])
			{
				m_count++;
			}
		}

		if (m_count >= 2)
		{
			return true;
		}
		return false;
	}

	bool acceptEqualPart(std::vector<int>& sample_n, std::vector<int>&  initial_holds_ids, std::vector<int>& iSample_desired_hold_ids)
	{
		for (unsigned int i = 0; i < iSample_desired_hold_ids.size(); i++)
		{
			if (iSample_desired_hold_ids[i] == sample_n[i] && initial_holds_ids[i] != sample_n[i])
			{
				return true;
			}

		}
		return false;
	}

	bool acceptDirectionLegsAndHands(Vector3 mPoint, std::vector<int>& sample_n_ids, std::vector<Vector3>& sample_n_points)
	{
		if (sample_n_ids[0] != -1 || sample_n_ids[1] != -1)
		{
			float z = 0;
			if (sample_n_ids[2] != -1)
			{
				z = std::max<float>(z, sample_n_points[2].z());
			}
			if (sample_n_ids[3] != -1)
			{
				z = std::max<float>(z, sample_n_points[3].z());
			}

			if (sample_n_ids[0] != -1)
			{
				if (z < sample_n_points[0].z())
				{
					return false;
				}
			}
			if (sample_n_ids[1] != -1)
			{
				if (z < sample_n_points[1].z())
				{
					return false;
				}
			}
		}

		return true;
	}

	//////////////////////////////////////////////////// general use for sampler ////////////////////////////////////////////////////

	bool isAllowedHandsLegsInDSample(std::vector<int>& initial_hold_ids, std::vector<int>& sampled_desired_hold_ids, bool isStanceGiven)
	{
		static std::vector<int> sample_initial_id;  //static to prevent heap alloc (a quick hack)
		sample_initial_id.clear();
		if (isStanceGiven)
		{
			if (initial_hold_ids[2] == -1 && initial_hold_ids[3] == -1)
			{
				if (sampled_desired_hold_ids[2] != -1 || sampled_desired_hold_ids[3] != -1)
				{
					return true;
				}
			}
			else if ((initial_hold_ids[2] != -1 || initial_hold_ids[3] != -1) && initial_hold_ids[0] == -1 && initial_hold_ids[1] == -1)
			{
				if (initial_hold_ids[2] == -1 && sampled_desired_hold_ids[2] != -1)
				{
					return true;
				}
				if (initial_hold_ids[3] == -1 && sampled_desired_hold_ids[3] != -1)
				{
					return true;
				}
				if (sampled_desired_hold_ids[0] != -1 || sampled_desired_hold_ids[1] != -1)
				{
					return true;
				}
			}
		}

		for (unsigned int i = 0; i < initial_hold_ids.size(); i++)
		{
			if (sampled_desired_hold_ids[i] == initial_hold_ids[i])
			{
				sample_initial_id.push_back(initial_hold_ids[i]);
			}
			else
			{
				sample_initial_id.push_back(-1);
			}
		}

		for (unsigned int i = 0; i < sample_initial_id.size(); i++)
		{
			if (sample_initial_id[i] == -1)
			{
				if (!isAllowedToReleaseHand_i(sample_initial_id, i) || !isAllowedToReleaseLeg_i(sample_initial_id, i))
					return false;
			}
		}
		return true;
	}

	bool isAllowedToReleaseHand_i(std::vector<int>& sampled_rInitial_hold_ids, int i)
	{
		if (i == sampled_rInitial_hold_ids.size() - 1 && sampled_rInitial_hold_ids[i - 1] == -1)
		{
			return false;
		}

		if (i == sampled_rInitial_hold_ids.size() - 2 && sampled_rInitial_hold_ids[i + 1] == -1)
		{
			return false;
		}

		if (sampled_rInitial_hold_ids[0] == -1 && sampled_rInitial_hold_ids[1] == -1 && i >= 2) // check feet
		{
			return false;
		}
		return true;
	}

	bool isAllowedToReleaseLeg_i(std::vector<int>& sampled_rInitial_hold_ids, int i)
	{
		if (sampled_rInitial_hold_ids[sampled_rInitial_hold_ids.size() - 1] != -1 && sampled_rInitial_hold_ids[sampled_rInitial_hold_ids.size() - 2] != -1)
		{
			return true;
		}

		if (i == 1 && sampled_rInitial_hold_ids[i - 1] == -1)
		{
			return false;
		}

		if (i == 0 && sampled_rInitial_hold_ids[i + 1] == -1)
		{
			return false;
		}

		return true;
	}

	void getAllHoldsInRangeAroundAgent(std::vector<int>& from_hold_ids, std::vector<std::vector<int>>& workSpaceHoldIDs)
	{

		std::vector<int> sampled_rInitial_hold_ids = from_hold_ids;

		for (unsigned int i = 0; i < sampled_rInitial_hold_ids.size(); i++)
		{
			std::vector<int> workSpaceHolds = getWorkSpaceAround(mOptCPBP::ControlledPoses(mOptCPBP::ControlledPoses::LeftLeg + i), from_hold_ids);
			workSpaceHoldIDs.push_back(workSpaceHolds);
		}

	}

	void getAllHoldsInRangeAroundAgent(mNode* iNode, std::vector<std::vector<size_t>>& workSpaceHoldIDs
		, std::vector<Vector3>& workspacePoints, std::vector<Vector3>& workSpacePointColor, std::vector<bool>& isIDInWorkSapceVector)
	{

		std::vector<int> sampled_rInitial_hold_ids = iNode->cNodeInfo.hold_bodies_ids;

		std::vector<Vector3> colors;
		colors.push_back(Vector3(255, 0, 0) / 255.0f); // blue cv::Scalar(255, 0, 0)
		colors.push_back(Vector3(0, 255, 0) / 255.0f); // green cv::Scalar(0, 255, 0)
		colors.push_back(Vector3(0, 0, 255) / 255.0f); // red cv::Scalar(0, 0, 255)
		colors.push_back(Vector3(255, 255, 255) / 255.0f); // whilte cv::Scalar(255, 255, 255)

		for (unsigned int i = 0; i < sampled_rInitial_hold_ids.size(); i++)
		{
			bool isIDInWorkSapce = false;
			std::vector<size_t> workSpaceHolds = getWorkSpaceAround(mOptCPBP::ControlledPoses(mOptCPBP::ControlledPoses::LeftLeg + i), iNode, isIDInWorkSapce);
			isIDInWorkSapceVector.push_back(isIDInWorkSapce);
			workSpaceHoldIDs.push_back(workSpaceHolds);
			
			for (unsigned int j = 0; j < workSpaceHolds.size(); j++)
			{
				Vector3 hold_pos_j = getHoldPos(workSpaceHolds[j]);
				workspacePoints.push_back(hold_pos_j);
				workSpacePointColor.push_back(colors[i]);
			}
		}

	}

	std::vector<int> getWorkSpaceAround(mOptCPBP::ControlledPoses iPosSource, std::vector<int>& from_hold_ids)
	{
		float max_radius_search = climberRadius / 1.3f; //2.6f * boneLength;
		float min_radius_search = climberRadius * 0.01f; //0.4f * boneLength;

		// evaluating starting pos
		std::vector<Vector3> from_hold_points; float size_d = 0;
		Vector3 trunk_pos = getHoldStancePosFrom(from_hold_ids, from_hold_points, size_d);
		trunk_pos[1] = 0;

		std::vector<size_t> holds_ids = getPointsInRadius(trunk_pos, max_radius_search);

		std::vector<int> desired_holds_ids;
		for (unsigned int i = 0; i < holds_ids.size(); i++)
		{
			Vector3 Pos_i = getHoldPos(holds_ids[i]);

			bool flag_add_direction_hold = false;

			float m_angle = SimulationContext::getAngleBtwVectorsXZ(Vector3(1.0f, 0.0f, 0.0f), Pos_i - trunk_pos);


			if ((m_angle <= 0.3f * PI) && (m_angle >= -PI) || (m_angle >= 0.3f * PI)) 
			{
				if (iPosSource == mOptCPBP::ControlledPoses::LeftLeg || iPosSource == mOptCPBP::ControlledPoses::RightLeg)
				{
					flag_add_direction_hold = true;
				}
			}
			if ((m_angle >= -0.3f * PI) && (m_angle <= PI) || (m_angle >= -0.3f * PI)) 
			{
				if (iPosSource == mOptCPBP::ControlledPoses::LeftHand || iPosSource == mOptCPBP::ControlledPoses::RightHand)
				{
					flag_add_direction_hold = true;
				}
			}

			if (flag_add_direction_hold)
			{
				float dis = (Pos_i - trunk_pos).norm();
				if (dis < max_radius_search && dis > min_radius_search)
				{
					desired_holds_ids.push_back(holds_ids[i]);
				}
			}
		}

		return desired_holds_ids;
	}

	std::vector<size_t> getWorkSpaceAround(mOptCPBP::ControlledPoses iPosSource, mNode* iNode, bool &isIDInWorkSapce)
	{
		float max_radius_search = climberRadius / 1.9f; //2.6f * boneLength;
		float min_radius_search = climberRadius * 0.08f; //0.4f * boneLength;

		isIDInWorkSapce = false;
		int cID = iNode->cNodeInfo.hold_bodies_ids[iPosSource - mOptCPBP::ControlledPoses::LeftLeg];

		// evaluating starting pos
		Vector3 sPos = iNode->cNodeInfo.getEndPointPosBones(SimulationContext::BodyName::BodyTrunk);
		sPos[1] = 0;

		std::vector<size_t> holds_ids = getPointsInRadius(sPos, max_radius_search);

		Vector3 trunk_pos = iNode->cNodeInfo.getEndPointPosBones(SimulationContext::BodyName::BodyTrunk);
		trunk_pos[1] = 0;
		Vector3 trunkDir = iNode->cNodeInfo.getBodyDirectionZ(SimulationContext::BodyName::BodyTrunk);//(SimulationContext::BodyName::BodyTrunkUpper) - trunk_pos;
		trunkDir[1] = 0;
		trunkDir = trunkDir.normalized();

		std::vector<size_t> desired_holds_ids;
		for (unsigned int i = 0; i < holds_ids.size(); i++)
		{
			Vector3 Pos_i = getHoldPos(holds_ids[i]);

			bool flag_add_direction_hold = false;
			float angle_btw_trunk = SimulationContext::getAngleBtwVectorsXZ(Pos_i - sPos, trunkDir); // we have a limit for turning direction with relation to trunk posture
			
			float angle_btw = angle_btw_trunk;

			float angle_btw_l = SimulationContext::getAngleBtwVectorsXZ(Vector3(1.0f, 0.0f, 0.0f), Pos_i - trunk_pos);
			switch (iPosSource)
			{
			case mOptCPBP::LeftLeg:
				if (angle_btw >= -PI && angle_btw <= -PI * 0.05f)
					flag_add_direction_hold = true;

				if (angle_btw <= PI && angle_btw > PI * 0.75f)
					flag_add_direction_hold = true;

				if (!(angle_btw_l < 0 && angle_btw_l > -PI))
					flag_add_direction_hold = false;
				break;
			case mOptCPBP::RightLeg:
				if (angle_btw >= PI * 0.05f && angle_btw <= PI)
					flag_add_direction_hold = true;

				if (angle_btw >= -PI && angle_btw <= -PI * 0.75f)
					flag_add_direction_hold = true;

				if (!(angle_btw_l < 0 && angle_btw_l > -PI))
					flag_add_direction_hold = false;
				break;
			case mOptCPBP::LeftHand:
				if (angle_btw >= -PI * 0.75f && angle_btw <= 0)
					flag_add_direction_hold = true;
				if (angle_btw >= 0 && angle_btw <= PI * 0.25f)
					flag_add_direction_hold = true;
				break;
			case mOptCPBP::RightHand:
				if (angle_btw >= 0 && angle_btw <= PI * 0.75f)
					flag_add_direction_hold = true;
				if (angle_btw >= -PI * 0.25f && angle_btw <= 0)
					flag_add_direction_hold = true;
				break;
			default:
				break;
			}

			if (flag_add_direction_hold)
			{
				float dis = (Pos_i - sPos).norm();
				if (dis < max_radius_search && dis > min_radius_search)
				{
					if (cID == holds_ids[i])
					{
						isIDInWorkSapce = true;
					}
					desired_holds_ids.push_back(holds_ids[i]);
				}
			}
		}

		return desired_holds_ids;
	}

	void setDesiredPosForOptimization(std::vector<mOptCPBP::ControlledPoses>& sourceP, std::vector<Vector3>& destinationP
		, std::vector<Vector3>& iRandDesPos, std::vector<int>& rand_desired_HoldsIds, bool flag_set_trunk)
	{
		if (flag_set_trunk)
			sourceP.push_back(mOptCPBP::ControlledPoses::MiddleTrunk);

		if (flag_set_trunk)
			destinationP.push_back(iRandDesPos[0]); // middle trunk's pos

		for (unsigned int i = 0; i < rand_desired_HoldsIds.size(); i++)
		{
			if (rand_desired_HoldsIds[i] != -1)
			{
				sourceP.push_back((mOptCPBP::ControlledPoses)(mOptCPBP::ControlledPoses::LeftLeg + i));
				if (flag_set_trunk)
					destinationP.push_back(iRandDesPos[i + 1]); // pos (first one is for trunk pos)
				else
					destinationP.push_back(iRandDesPos[i]); // pos 
			}
		}

		return;
	}

	/////////////////////////////////////////// handling holds using kd-tree ///////////////////////////////////////////////////////
	
	std::vector<double> getHoldKey(Vector3& c)
	{
		std::vector<double> rKey;

		rKey.push_back(c.x());
		rKey.push_back(c.y());
		rKey.push_back(c.z());

		return rKey;
	}
public:

	/////////////////////////////////////////// handling holds using kd-tree ///////////////////////////////////////////////////////

	int getClosestPoint(Vector3& qP)
	{
		std::vector<int> ret_index = mHoldsKDTree.nearest(getHoldKey(qP), 1);

		if (ret_index.size() > 0)
			return ret_index[0];
		return -1;
	}

	std::vector<size_t> getPointsInRadius(Vector3& qP, float r)
	{
		std::vector<size_t> ret_index;
		for (unsigned int i = 0; i < myHoldPoints.size(); i++)
		{
			Vector3 hold_i = myHoldPoints[i];
			float cDis = (hold_i - qP).norm();
			if (cDis < r)
			{
				ret_index.push_back(i);
			}
		}
		return ret_index;
	}

}* mHoldSampler;

class mStanceNode
{
public:
	bool isItExpanded; // for building the graph
	std::vector<int> hold_ids; // ll, rl, lh, rh

	int stanceIndex;
	std::vector<int> parentStanceIds; // parent Nodes
	std::vector<bool> isItTried_to_father;

	std::vector<int> childStanceIds; // childNodes

	float nodeCost; // cost of standing at node of graph
	std::vector<float> cost_transition_to_child;
	std::vector<float> cost_moveLimb_to_child;
	std::vector<bool> isItTried_to_child;

	float g_AStar;
	float h_AStar;

	bool dijkstraVisited;

//	int count_u;
	int childIndexInbFather;
	int bFatherStanceIndex;
	int bChildStanceIndex;

	mStanceNode(std::vector<int>& iInitStance)
	{
		g_AStar = FLT_MAX;
		h_AStar = FLT_MAX;
		bFatherStanceIndex = -1;
		bChildStanceIndex = -1;
		childIndexInbFather = -1;
//		count_u = 0;

		hold_ids = iInitStance;
		isItExpanded = false;
		stanceIndex = -1;
	}
};

class mStancePathNode // for A* prune
{
public:
	mStancePathNode(std::vector<int>& iPath, float iG, float iH)
	{
		_path.reserve(100);

		mG = iG;
		mH = iH;
		for (unsigned int i = 0; i < iPath.size(); i++)
		{
			_path.push_back(iPath[i]);
		}
	}

	std::vector<int> addToEndPath(int stanceID)
	{
		std::vector<int> nPath;
		for (unsigned int i = 0; i < _path.size(); i++)
		{
			nPath.push_back(_path[i]);
		}
		nPath.push_back(stanceID);
		return nPath;
	}

	std::vector<int> _path;
	float mG;
	float mH;
};


static inline uint32_t stanceToKey(const std::vector<int>& stance)
{
	uint32_t result=0;
	for (int i=0; i<4; i++)
	{
		uint32_t uNode=(uint32_t)(stance[i]+1);  //+1 to handle the -1
		if (!(uNode>=0 && uNode<256))
			Debug::throwError("Invalid hold index %d!\n",uNode);
		result=(result+uNode);
		if (i<3)
			result=result << 8; //shift, assuming max 256 holds
	}
	return result;
}


class mStanceGraph
{
	std::vector<int> initialStance;
	float _MaxTriedTransitionCost;
public:
	enum mAlgSolveGraph{myAlgAStar = 0, AStarPrune = 1, myAStarPrune = 2, myDijkstraHueristic = 3};

	std::vector<std::vector<int>> m_found_paths;
	std::unordered_map<uint32_t, int> stanceToNode; //key is uint64_t computed from a stance (i.e., a vector of 4 ints), data is index to the stance nodes vector
	DynamicMinMax minmax;

	mStanceGraph()
	{
		maxGraphDegree = 0;
		stanceToNode.rehash(1000003);  //a prime number of buckets to minimize different keys mapping to same bucket
		mClimberSampler = nullptr;

		_MaxTriedTransitionCost = 100000.0f;
		//preVal = -FLT_MAX; // for debugging
		initializeOpenListAStarPrune();
	}

	void initializeOpenListAStarPrune()
	{
		openListPath.clear();
		std::vector<int> rootPath; rootPath.push_back(0);
		openListPath.push_back(mStancePathNode(rootPath,0,0));
	}

	float getGUpdatedCostNode(int _tNode)
	{
		bool flag_continue = true;
		int cNode = _tNode;
		float tCost = 0.0f;
		while (flag_continue)
		{
			int bFather = stance_nodes[cNode].bFatherStanceIndex;
			if (bFather != -1)
			{
				mStanceNode& fNode = stance_nodes[bFather];
				int child_index = getChildIndex(bFather,cNode);
				float cCost = fNode.cost_transition_to_child[child_index] 
							+ fNode.cost_moveLimb_to_child[child_index] 
							+ stance_nodes[child_index].nodeCost;
				tCost += cCost;
				cNode = bFather;
			}
			else
			{
				flag_continue = false;
			}
		}
		return tCost;
	}

	float getHeuristicCostNodeToNode(int _fNode, int _tNode)
	{
		mStanceNode& fNode = stance_nodes[_fNode];
		mStanceNode& tNode = stance_nodes[_tNode];
		int c = getChildIndex(_fNode, _tNode);
		float nH = tNode.h_AStar + tNode.nodeCost + fNode.cost_transition_to_child[c] 
				+ fNode.cost_moveLimb_to_child[c] + getCostToGoal(fNode.hold_ids);
		return nH;
	}

	void updateHeuristicsForNode(int _fNode, int _tNode)
	{
		std::vector<int> mUpdateList;
		mUpdateList.reserve(stance_nodes.size() + 1);

		//
		mUpdateList.push_back(_fNode);
		while (mUpdateList.size() > 0)
		{
			int cNode = mUpdateList[0];
			mUpdateList.erase(mUpdateList.begin());

			mStanceNode& cStanceNode = stance_nodes[cNode];
			float min_val = FLT_MAX;
			int min_index = -1;
			for (unsigned int c = 0; c < cStanceNode.childStanceIds.size(); c++)
			{
				mStanceNode& childStanceNode = stance_nodes[cStanceNode.childStanceIds[c]];
				float cCost = getHeuristicCostNodeToNode(cNode, cStanceNode.childStanceIds[c]);
				if (min_val > cCost)
				{
					min_val = cCost;
					min_index = c;
				}
			}
			cStanceNode.bChildStanceIndex = cStanceNode.childStanceIds[min_index];
			cStanceNode.h_AStar = min_val;

			for (unsigned int f = 0; f < cStanceNode.parentStanceIds.size(); f++)
			{
				mStanceNode& fatherStanceNode = stance_nodes[cStanceNode.parentStanceIds[f]];
				if (fatherStanceNode.bChildStanceIndex == cNode)
				{
					mUpdateList.push_back(cStanceNode.parentStanceIds[f]);
				}
			}
		}
		// update best father index

	}

	/////////////////////////////////////////////////////////////////////////////////////////////////// build graph is not complete: needs debugging

	void buildGraph(std::vector<int>& iCStance, std::vector<int>& iInitStance) 
	{
		initialStance = iInitStance;

		if (mClimberSampler == nullptr)
		{
			return;
		}
		if (mClimberSampler->myHoldPoints.size() == 0)
		{
			return;
		}
		mRootGraph = addGraphNode(-1, iCStance);
		
		std::vector<int> expandNodes;
		expandNodes.push_back(mRootGraph);

		//add all nodes connected to initial stance.
		//PERTTU: Todo: what do we need the while for? Isn't it enough to add the mInitialList once, as getListOfStanceSamples() always gets the iInitStance as argument?
		while(expandNodes.size() > 0)
		{
			int exanpd_stance_id = expandNodes[0];
			expandNodes.erase(expandNodes.begin());
			stance_nodes[exanpd_stance_id].isItExpanded = true;

			std::vector<std::vector<int>> mInitialList = mClimberSampler->getListOfStanceSamples(stance_nodes[exanpd_stance_id].hold_ids, iInitStance, true);

			for (unsigned int i = 0; i < mInitialList.size(); i++)
			{
				int sId = addGraphNode(exanpd_stance_id, mInitialList[i]);

				if (!stance_nodes[sId].isItExpanded && !mSampler::isInSetHoldIDs(sId, expandNodes))
				{
					expandNodes.push_back(sId);
				}
			}
		}
		
		/*expandNodes.push_back(findStanceFrom(iInitStance));

		while(expandNodes.size() > 0)
		{
			int exanpd_stance_id = expandNodes[0];
			expandNodes.erase(expandNodes.begin());
			stance_nodes[exanpd_stance_id].isItExpanded = true;

			std::vector<std::vector<int>> all_possible_stances = mClimberSampler->getListOfStanceSamplesAround(stance_nodes[exanpd_stance_id].hold_ids, false);

			for (unsigned int ws = 0; ws < all_possible_stances.size(); ws++)
			{
				std::vector<std::vector<int>> mInitialList = mClimberSampler->getListOfStanceSamples(stance_nodes[exanpd_stance_id].hold_ids, all_possible_stances[ws], false);

				for (unsigned int i = 0; i < mInitialList.size(); i++)
				{
					int sId = addGraphNode(exanpd_stance_id, mInitialList[i]);

					if (!stance_nodes[sId].isItExpanded && !mSampler::isInSetHoldIDs(sId, expandNodes))
					{
						expandNodes.push_back(sId);
					}
				}
			}
		}*/

		int fStance = findStanceFrom(iInitStance);

		//std::vector<int> dStance;
		//dStance.push_back(1);
		//dStance.push_back(1);
		//dStance.push_back(2);
		//dStance.push_back(3);

		expandNodes.push_back(fStance);
		std::vector<int> reachable_holds_ids;
		

		//add other nodes
		while(expandNodes.size() > 0)
		{
			int exanpd_stance_id = expandNodes[0];
			expandNodes.erase(expandNodes.begin());
			stance_nodes[exanpd_stance_id].isItExpanded = true;

			for (unsigned int i = 0; i < stance_nodes[exanpd_stance_id].hold_ids.size(); i++)
			{
				std::vector<int> higher_stance_ids;
				if (stance_nodes[exanpd_stance_id].hold_ids[i] != -1)
				{
					if (i <= 1)
					{
						// for feet add all reachable holds above them
						higher_stance_ids = mClimberSampler->indices_higher_than[stance_nodes[exanpd_stance_id].hold_ids[i]];
					}
					else
					{
						// add hands for finding all combinations under them
						mSampler::addToSetHoldIDs(stance_nodes[exanpd_stance_id].hold_ids[i], reachable_holds_ids);
					}
				}
				for (unsigned int m = 0; m < higher_stance_ids.size(); m++)
				{
					mSampler::addToSetHoldIDs(higher_stance_ids[m], reachable_holds_ids);
				}
			}

			std::vector<std::vector<int>> all_possible_stances;
			for (unsigned int j = 0; j < reachable_holds_ids.size(); j++)
			{
				int hold_id_i = reachable_holds_ids[j];

				mClimberSampler->getListOfStanceSamplesAround(hold_id_i, stance_nodes[exanpd_stance_id].hold_ids,all_possible_stances);

				for (unsigned int ws = 0; ws < all_possible_stances.size(); ws++)
				{
					/*if (mNode::isSetAEqualsSetB(dStance, all_possible_stances[ws]) && fStance == exanpd_stance_id)
					{
						int notifyme = 1;
					}*/

					//std::vector<std::vector<int>> mInitialList = mClimberSampler->getListOfStanceSamples(stance_nodes[exanpd_stance_id].hold_ids, all_possible_stances[ws], false);

					//for (unsigned int i = 0; i < mInitialList.size(); i++)
					//{
					int sId = addGraphNode(exanpd_stance_id, all_possible_stances[ws]);//mInitialList[i]);

					if (!stance_nodes[sId].isItExpanded && !mSampler::isInSetHoldIDs(sId, expandNodes))
					{
						expandNodes.push_back(sId);
					}
					//}
				}
			}
			reachable_holds_ids.clear();
		}

//		applyDijkstraAll2(); // used for admisibale A*
		minmax.init(stance_nodes.size());
		printf("%d", stance_nodes.size());

		//for (volatile int j=0; j<10000; j++)
		//	applyDijkstraAll2();

		return;
	}

	bool updateGraphNN(int _fNode, int _tNode)
	{
		for (unsigned int i = 0; i < stance_nodes[_fNode].childStanceIds.size(); i++)
		{
			if (stance_nodes[_fNode].childStanceIds[i] == _tNode)
			{
				stance_nodes[_fNode].cost_transition_to_child[i] += _MaxTriedTransitionCost;
//				updateHeuristicsForNode(_fNode, _tNode);
				return true;
			}
		}
		return false;
	}
	
	std::vector<int> solveGraph(float& ret_min_val, mAlgSolveGraph algID)
	{
		std::vector<int> ret_path;
		switch (algID)
		{
		case mAlgSolveGraph::myAlgAStar:
			ret_path = solveAStar(ret_min_val);
			break;
		case mAlgSolveGraph::AStarPrune:
			ret_path = solveAStarPrune(ret_min_val);
			break;
		case mAlgSolveGraph::myAStarPrune:
			ret_path = solveMYAStarPrune(ret_min_val);
			break;
		case mAlgSolveGraph::myDijkstraHueristic:
			ret_path = solveAStarPrune2(ret_min_val);
			break;
		default:
			ret_path = solveAStar(ret_min_val);
			break;
		}
		return ret_path;
	}

	std::vector<std::vector<int>> returnPath()
	{
		return returnPathFrom(retPath);
	}

	int findStanceFrom(const std::vector<int>& _fStance)
	{
		uint32_t key=stanceToKey(_fStance);
		std::unordered_map<uint32_t,int>::iterator it=stanceToNode.find(key);
		if (it==stanceToNode.end())
			return -1;
		return it->second;
		//int stance_id = -1;
		//for (unsigned int i = 0; i < stance_nodes.size() && stance_id == -1; i++)
		//{
		//	if (mNode::isSetAEqualsSetB(stance_nodes[i].hold_ids, _fStance))
		//	{
		//		stance_id = i;
		//	}
		//}
		//return stance_id;
	}

	std::vector<int> getStanceGraph(int _graphNodeId)
	{
		return stance_nodes[_graphNodeId].hold_ids;
	}

	int getIndexStanceNode(int indexPath)
	{
		int j = retPath.size() - 1 - indexPath;
		if (j < (int)retPath.size())
		{
			return retPath[j];
		}
		return mRootGraph;
	}

	void setCurPathTriedFather()
	{
		for (unsigned int i = 0; i < retPath.size(); i++)
		{
			setTriedToCurrentFather(retPath[i], stance_nodes[retPath[i]].bFatherStanceIndex, true);
			setTriedToCurrentChild(stance_nodes[retPath[i]].bFatherStanceIndex, retPath[i], true);
		}
		return;
	}

	std::vector<int> retPath;

	std::list<mStancePathNode> openListPath;

	mSampler* mClimberSampler;

	Vector3 goalPos;
private:
	///////////////////////////// my alg A* ////////////////////////////////////
	std::vector<int> solveAStar(float& ret_min_val)
	{
		std::vector<int> openList;
		std::vector<int> closeList;

		stance_nodes[mRootGraph].g_AStar = 0;
		stance_nodes[mRootGraph].h_AStar = 0;

		openANode(mRootGraph, openList, closeList);

		int sIndex = findStanceFrom(initialStance);

		bool flag_continue = true;
		while (flag_continue)
		{
			int openIndexi = stanceIndexLowestFValue(openList, ret_min_val);

			if (isGoalFound(openIndexi))
			{
				retPath = returnPathFrom(openIndexi);
				if (mSampler::isInSetHoldIDs(sIndex, retPath))
				{
					if (!mSampler::isInSampledStanceSet(retPath, m_found_paths))
					{
						flag_continue = false;
					}
					else
					{
						mSampler::removeFromSetHoldIDs(openIndexi, openList);
						continue;
					}
				}
				else
				{
					retPath.clear();
				}
			}

			if (flag_continue && openIndexi >= 0)
			{
				openANode(openIndexi, openList, closeList);
			}

			if (openList.size() == 0)
			{
				flag_continue = false;
			}
		}

		/*if (preRetPath.size() == 0 || preVal <= ret_min_val)
		{
			preRetPath = retPath;
			preVal = ret_min_val;
		}*/
		/*else
		{
			float v;
			solveAStar(v);
		}*/
		return retPath;
	}

	void openANode(int gNodei, std::vector<int>& openList, std::vector<int>& closeList)
	{
		mSampler::removeFromSetHoldIDs(gNodei, openList);
		mSampler::addToSetHoldIDs(gNodei, closeList);

		mStanceNode n_i = stance_nodes[gNodei];
		for (unsigned int i = 0; i < n_i.childStanceIds.size(); i++)
		{
			int cIndex = n_i.childStanceIds[i];
			mStanceNode* n_child_i = &stance_nodes[cIndex];

			if (mSampler::isInSetHoldIDs(cIndex, closeList))
			{
				continue;
			}

			int oBestFather = n_child_i->bFatherStanceIndex;

			float nG = n_i.g_AStar + n_i.cost_transition_to_child[i] + n_i.cost_moveLimb_to_child[i] + n_child_i->nodeCost;
			float nH = getCostToGoal(n_child_i->hold_ids);
			float nF = nG + nH;

			if (!mSampler::isInSetHoldIDs(cIndex, openList))
			{
				float oG = n_child_i->g_AStar;

				n_child_i->g_AStar = nG;
				n_child_i->h_AStar = nH;
				n_child_i->bFatherStanceIndex = gNodei;
				n_child_i->childIndexInbFather = i;

				if (isLoopCreated(cIndex))
				{
					n_child_i->bFatherStanceIndex = oBestFather;
					n_child_i->g_AStar = oG;
				}

				mSampler::addToSetHoldIDs(cIndex, openList);
			}
			else
			{
				float oG = n_child_i->g_AStar;

				if (nG < oG || (isItTriedToCurrentFather(n_child_i->stanceIndex, n_child_i->bFatherStanceIndex) && isAllOfNodeChildrenTried(n_child_i->stanceIndex)))
				{
					n_child_i->g_AStar = nG;
					n_child_i->h_AStar = nH;
					n_child_i->bFatherStanceIndex = gNodei;
					n_child_i->childIndexInbFather = i;

					if (isLoopCreated(cIndex))
					{
						n_child_i->bFatherStanceIndex = oBestFather;
						n_child_i->g_AStar = oG;
					}
				}
			}
		}

		return;
	}

	int stanceIndexLowestFValue(std::vector<int>& openList, float& ret_min_val)
	{
		float minVal_notTried = FLT_MAX;
		int min_index_notTried = -1;

		float minVal = FLT_MAX;
		int min_index = -1;
		for (unsigned int i = 0; i < openList.size(); i++)
		{
			//mStanceNode* cFather = nullptr;
			//if (stance_nodes[openList[i]].bFatherStanceIndex != -1)
			//{
			//	cFather = &stance_nodes[stance_nodes[openList[i]].bFatherStanceIndex];
			//}
			mStanceNode* cNode = &stance_nodes[openList[i]];

			float cVal = stance_nodes[openList[i]].g_AStar + stance_nodes[openList[i]].h_AStar;

			if (cVal < minVal_notTried && (!isItTriedToCurrentFather(cNode->stanceIndex, cNode->bFatherStanceIndex) || !isAllOfNodeChildrenTried(openList[i])))
			{
				minVal_notTried = cVal;
				min_index_notTried = openList[i];
			}

			if (cVal < minVal)
			{
				minVal = cVal;
				min_index = openList[i];
			}
		}

		ret_min_val = minVal_notTried;

		if (min_index_notTried >= 0)
		{
			return min_index_notTried;
		}

		ret_min_val = minVal;
		return min_index;
	}

	///////////////////////////// A*prune /////////////////////////////////////

	void applyDijkstra(int fIndex, int sIndex)
	{
		for (unsigned int i = 0; i < stance_nodes.size(); i++)
		{
			stance_nodes[i].g_AStar = FLT_MAX;
		}

		std::vector<int> openList;
		std::vector<int> closeList;

		stance_nodes[fIndex].g_AStar = 0;
		openList.push_back(fIndex);
		
		while (openList.size() > 0)
		{
			mStanceNode* cNode = &stance_nodes[openList[0]];
			openList.erase(openList.begin());
			mSampler::addToSetHoldIDs(cNode->stanceIndex, closeList);

			if (isGoalFound(cNode->stanceIndex))
			{
				retPath = returnPathFrom(cNode->stanceIndex);
				stance_nodes[fIndex].h_AStar = cNode->g_AStar;
				return;
			}

			for (unsigned int c = 0; c < cNode->childStanceIds.size(); c++)
			{
				mStanceNode* ccNode = &stance_nodes[cNode->childStanceIds[c]];
				float nG = cNode->g_AStar + cNode->cost_transition_to_child[c] + cNode->cost_moveLimb_to_child[c] + ccNode->nodeCost;
				if (nG < ccNode->g_AStar)
				{
					ccNode->g_AStar = nG;
					ccNode->bFatherStanceIndex = cNode->stanceIndex;
				}

				if (!mSampler::isInSetHoldIDs(ccNode->stanceIndex, closeList))
				{
					if (mSampler::isInSetHoldIDs(ccNode->stanceIndex, openList))
					{
						mSampler::removeFromSetHoldIDs(ccNode->stanceIndex, openList);
					}
					unsigned int i = 0;
					for (i = 0; i < openList.size(); i++)
					{
						float cF = stance_nodes[openList[i]].g_AStar;
						float nF = nG;
						if (nF < cF)
						{
							break;
						}
					}
					openList.insert(openList.begin() + i, ccNode->stanceIndex);
				}
			}
		}

		return;
	}

	std::vector<int> solveAStarPrune(float& ret_min_val)
	{
		int sIndex = findStanceFrom(initialStance);
		for (unsigned int i = 0; i < stance_nodes.size(); i++)
		{
			if (!isGoalFound(stance_nodes[i].stanceIndex))
				applyDijkstra(stance_nodes[i].stanceIndex, sIndex);
			else
				stance_nodes[i].h_AStar = 0.0f;
		}

		std::vector<mStancePathNode> openListPath;

		std::vector<int> rootPath; rootPath.push_back(mRootGraph);
		openListPath.push_back(mStancePathNode(rootPath,0,0));

		bool flag_continue = true;
		while (flag_continue)
		{
			mStancePathNode fNode = openListPath[0];
			openListPath.erase(openListPath.begin());

			int eStanceIndex = fNode._path[fNode._path.size()-1];

			if (isGoalFound(eStanceIndex))
			{
				retPath = reversePath(fNode._path);
				ret_min_val = fNode.mG + fNode.mH;
				if (mSampler::isInSetHoldIDs(sIndex, retPath))
				{
					if (!mSampler::isInSampledStanceSet(retPath, m_found_paths))
					{
						flag_continue = false;
					}
				}
				else
				{
					retPath.clear();
				}
			}

			if (flag_continue)
			{
				mStanceNode stanceNode = stance_nodes[eStanceIndex];
				for (unsigned int c = 0; c < stanceNode.childStanceIds.size(); c++)
				{
					mStanceNode stanceChild = stance_nodes[stanceNode.childStanceIds[c]];
				
					if (mSampler::isInSetHoldIDs(stanceChild.stanceIndex, fNode._path)) // no loop
						continue;

					float nG = fNode.mG + stanceNode.cost_transition_to_child[c] + stanceNode.cost_moveLimb_to_child[c] + stanceChild.nodeCost;
					float nH = stanceChild.h_AStar;

					std::vector<int> nPath = fNode.addToEndPath(stanceChild.stanceIndex);

					mStancePathNode nStancePath(nPath, nG, nH);

					insertNodePath(openListPath, nStancePath);
				}
			}

			if (openListPath.size() == 0)
			{
				flag_continue = false;
			}
		}

		return retPath;
	}

	void insertNodePath(std::vector<mStancePathNode>& openList, mStancePathNode& nNode)
	{
		unsigned int i = 0;
		for (i = 0 ; i < openList.size(); i++)
		{
			mStancePathNode n_i = openList[i];
			float cF = n_i.mG + n_i.mH;
			float nF = nNode.mG + nNode.mH;
			if (nF < cF)
			{
				break;
			}
		}
		openList.insert(openList.begin() + i, nNode);
		return;
	}

	////////////////////////////////////////////////////////////// A* prune my version

	std::vector<int> solveMYAStarPrune(float& ret_min_val)
	{
		int max_count = 200000;
		int sIndex = findStanceFrom(initialStance);

		std::list<mStancePathNode> openListPath;
		std::list<mStancePathNode> closeListPath;

		std::vector<int> openList; openList.reserve(max_count);
		std::vector<int> closeList; closeList.reserve(max_count);

		std::vector<int> rootPath; rootPath.push_back(mRootGraph);
		openList.push_back(mRootGraph);
		openListPath.push_back(mStancePathNode(rootPath,0,0));

		bool flag_continue = true;
		while (flag_continue)
		{
			if (openListPath.size() == 0)
			{
				movePathsFromTo(closeListPath, openListPath, openList, closeList);
			}

			if (openListPath.size() == 0)
			{
				break;
			}

			mStancePathNode fNode = openListPath.front();
			openListPath.pop_front();// erase(openListPath.begin());

			int eStanceIndex = fNode._path[fNode._path.size()-1];

			if (isGoalFound(eStanceIndex))
			{
				retPath = reversePath(fNode._path);
				ret_min_val = fNode.mG + fNode.mH;
				if (mSampler::isInSetHoldIDs(sIndex, retPath))
				{
					if (!mSampler::isInSampledStanceSet(retPath, m_found_paths))
					{
						flag_continue = false;
					}
					else
					{
						mSampler::removeFromSetHoldIDs(eStanceIndex, openList);
						continue;
					}
				}
				else
				{
					retPath.clear();
				}
			}

			if (flag_continue)
			{
				mStanceNode& stanceNode = stance_nodes[eStanceIndex];

				mSampler::removeFromSetHoldIDs(eStanceIndex, openList);
				mSampler::addToSetHoldIDs(eStanceIndex, closeList);

				for (unsigned int c = 0; c < stanceNode.childStanceIds.size(); c++)
				{
					mStanceNode& stanceChild = stance_nodes[stanceNode.childStanceIds[c]];
				
					if (mSampler::isInSetHoldIDs(stanceChild.stanceIndex, fNode._path)) // no loop
						continue;

					float nG = fNode.mG + stanceNode.cost_transition_to_child[c] + stanceNode.cost_moveLimb_to_child[c] + stanceChild.nodeCost;
					float nH = getCostToGoal(stanceChild.hold_ids);

					std::vector<int> nPath = fNode.addToEndPath(stanceChild.stanceIndex);

					mStancePathNode nStancePath(nPath, nG, nH);

					insertNodePath(openListPath, closeListPath, nStancePath, openList, closeList);
				}
			}

			if (openListPath.size() == 0 && closeListPath.size() == 0)
			{
				flag_continue = false;
			}
		}

		return retPath;
	}

	void movePathsFromTo(std::list<mStancePathNode>& closeListPath, std::list<mStancePathNode>& openListPath, std::vector<int>& openListID, std::vector<int>& closeListID)
	{
		unsigned int max_open_list_size = 100;
		std::list<mStancePathNode>::iterator iter_pathNode = closeListPath.begin();
		int max_counter = closeListPath.size();
		closeListID.clear();
		for (int i = 0; i < max_counter; i++)
		{
			const mStancePathNode& nPath = *iter_pathNode;
			int eIndex = nPath._path[nPath._path.size() - 1];

//			mSampler::removeFromSetHoldIDs(eIndex, closeListID);

			mStancePathNode nNode = *iter_pathNode;
			iter_pathNode = closeListPath.erase(iter_pathNode);
			insertNodePath(openListPath, closeListPath, nNode, openListID, closeListID);

			if (openListID.size() > max_open_list_size)
			{
				return;
			}

			if (iter_pathNode == closeListPath.end())
				break;
		}
		return;
	}

	void insertNodePath(std::list<mStancePathNode>& openListPath, std::list<mStancePathNode>& closeListPath, mStancePathNode& nNode, std::vector<int>& openListID, std::vector<int>& closeListID)
	{
		unsigned int fIndex = 0;
		unsigned int eIndex = 0;

		int cIndex = nNode._path[nNode._path.size()-1];
		if (mSampler::isInSetHoldIDs(cIndex, closeListID))
		{
			closeListPath.push_back(nNode);
			return;
		}
		else
		{
			if (!mSampler::isInSetHoldIDs(cIndex, openListID))
			{
				eIndex = openListID.size();
				mSampler::addToSetHoldIDs(cIndex, openListID);
			}
			else
			{
				
				std::list<mStancePathNode>::iterator iter_pathNode = openListPath.begin();
				for (unsigned int i = 0; i < openListID.size(); i++)
				{
					const mStancePathNode& n_i = *iter_pathNode;

					int endIndex = n_i._path[n_i._path.size()-1];
					if (endIndex == cIndex)
					{
						float cF = n_i.mG + n_i.mH;
						float nF = nNode.mG + nNode.mH;
						if (nF < cF)
						{
							mStancePathNode copy_n_i = *iter_pathNode;
							iter_pathNode = openListPath.erase(iter_pathNode);
							insertPathNode(openListPath, nNode, 0, openListID.size() - 1);
							closeListPath.push_back(copy_n_i);
							return;
						}
						else
						{
							closeListPath.push_back(nNode);
							return;
						}
					}
					iter_pathNode++;
				}
			}
			
		}
		insertPathNode(openListPath, nNode, fIndex, eIndex);
		return;
	}

	void insertPathNode(std::list<mStancePathNode>& openListPath, const mStancePathNode& nNode, unsigned int fIndex, unsigned int eIndex)
	{
		
		std::list<mStancePathNode>::iterator iter = openListPath.begin();

		//while (iter != openListPath.end()){
		//	//CODE HERE
		//	iter++;
		//}

		unsigned int i = fIndex;
		for (i = fIndex; i < eIndex; i++)
		{
			const mStancePathNode& n_i = *iter;
			float cF = n_i.mG + n_i.mH;
			float nF = nNode.mG + nNode.mH;
			if (nF < cF)
			{
				break;
			}
			iter++;
		}
		openListPath.insert(iter, nNode);
	}

	//////////////////////////////////////////////////////////////////

	int getChildIndex(int _fNode, int _tNode)
	{
		for (unsigned int i = 0; i < stance_nodes[_fNode].childStanceIds.size(); i++)
		{
			if (stance_nodes[_fNode].childStanceIds[i] == _tNode)
			{
				return i;
			}
		}
		return -1;
	}

	void applyDijkstraAll2()
	{
		//Run Dijkstra backwards, initializing all goal nodes to zero cost and others to infinity.
		//This results in each node having the cost as the minimum cost to go towards any of the goals.
		//This will then be used as the heuristic in A* prune. Note that if the climber is able to make
		//all the moves corresponding to all edges, the heuristic equals the actual cost, and A* will be optimal.
		//In case the climber fails, this function will be called again, i.e., the heuristic will be updated
		//after each failure.
		//Note: minmax is a DynamicMinMax instance - almost like std::priority_queue, but supports updating 
		//the priorities of elements at a logN cost without removal & insertion. 
		for (unsigned int i = 0; i < stance_nodes.size(); i++)
		{
			stance_nodes[i].g_AStar = FLT_MAX;   
			stance_nodes[i].h_AStar = FLT_MAX;  //this is the output value, i.e., the "heuristic" used by A* prune
			stance_nodes[i].dijkstraVisited=false;
			minmax.setValue(i,FLT_MAX);
		}
		//for (unsigned int i = 0; i < reached_goals.size(); i++)
		//{
		//	openList.push_back(reached_goals[i]);
		//	stance_nodes[reached_goals[i]].h_AStar = 0;
		//	minmax.setValue(reached_goals[i],0);
		//}
		for (unsigned int i = 0; i < goal_stances.size(); i++)
		{
			stance_nodes[goal_stances[i]].h_AStar = 0;
			minmax.setValue(goal_stances[i],0);
		}

	//	int sIndex = findStanceFrom(initialStance);
		int nVisited=0;
		while (nVisited<(int)stance_nodes.size())
		{
			//get the node with least cost (initially, all goal nodes have zero cost)
			mStanceNode* cNode =&stance_nodes[minmax.getMinIdx()];
			
			//loop over neighbors (parentStanceIds really contains all the neighbors)
			//and update their costs
			for (unsigned int f = 0; f < cNode->parentStanceIds.size(); f++)
			{
				mStanceNode* fNode = &stance_nodes[cNode->parentStanceIds[f]];
				int c = getChildIndex(cNode->parentStanceIds[f], cNode->stanceIndex);
				float nH = cNode->h_AStar + cNode->nodeCost + fNode->cost_transition_to_child[c] 
						+ fNode->cost_moveLimb_to_child[c];// + getDisFromStanceToStance(*fNode, *cNode);
							
				//getCostToGoal(fNode->hold_ids);  //TODO fix getCostToGoal! (should equal the minimum distance from left or right hand to goal, and the same should be used as the actual cost).
				
						/*float cDis = 0.0f;
				for (unsigned int h_i = 0; h_i < fNode->hold_ids.size(); h_i++)
				{
					if (fNode->hold_ids[h_i] != -1 && cNode->hold_ids[h_i] != -1)
					{
						cDis+=(mClimberSampler->getHoldPos(fNode->hold_ids[h_i]) - mClimberSampler->getHoldPos(cNode->hold_ids[h_i])).squaredNorm();
					}
				}
				nH += sqrt(cDis);*/


				//PERTTU: commented this check out - makes no sense. We should update all neighbors except the already visited ones
				//if (cNode->bFatherStanceIndex == cNode->parentStanceIds[f] || fNode->bFatherStanceIndex >= 0)
				if (!fNode->dijkstraVisited)
				{
					if (nH < fNode->h_AStar)
					{
						fNode->h_AStar = nH;
						fNode->bChildStanceIndex = cNode->stanceIndex;
						minmax.setValue(fNode->stanceIndex,nH);
					}
				}
			} //all neighbors
			//Mark the node as visited. Also set it's priority to FLT_MAX so that it will not be returned by minmax.getMinIdx().
			//Note that since dijkstraVisited is now true, the priority will not be updated again and will stay at FLT_MAX for the remainder of the algorithm.
			cNode->dijkstraVisited=true;
			minmax.setValue(cNode->stanceIndex,FLT_MAX);
			nVisited++;
		}
		
		return;
	}

	std::vector<int> solveAStarPrune2(float& ret_min_val)
	{
		int k = 100; // k shortest path in the paper
		int max_number_paths = max<int>(maxGraphDegree + 20, k);

		int sIndex = findStanceFrom(initialStance);

		applyDijkstraAll2();

		bool flag_continue = true;
		while (flag_continue)
		{
			mStancePathNode fNode = openListPath.front();
			openListPath.pop_front();// erase(openListPath.begin());

			int eStanceIndex = fNode._path[fNode._path.size()-1];

			if (isGoalFound(eStanceIndex))
			{
				retPath = reversePath(fNode._path);
				ret_min_val = fNode.mG + fNode.mH;
				if (mSampler::isInSetHoldIDs(sIndex, retPath))
				{
					if (!mSampler::isInSampledStanceSet(retPath, m_found_paths))
					{
						flag_continue = false;
					}
					else
					{
						continue;
					}
				}
				else
				{
					retPath.clear();
				}
			}

			if (flag_continue)
			{
				mStanceNode& stanceNode = stance_nodes[eStanceIndex];
				for (unsigned int c = 0; c < stanceNode.childStanceIds.size(); c++)
				{
					mStanceNode& stanceChild = stance_nodes[stanceNode.childStanceIds[c]];

					if (mSampler::isInSetHoldIDs(stanceChild.stanceIndex, fNode._path)) // no loop
						continue;

					float nG = fNode.mG + stanceNode.cost_transition_to_child[c] + stanceNode.cost_moveLimb_to_child[c] + stanceChild.nodeCost;// + getDisFromStanceToStance(stanceNode, stanceChild);
					float nH = stanceChild.h_AStar;

					std::vector<int> nPath = fNode.addToEndPath(stanceChild.stanceIndex);

					mStancePathNode nStancePath(nPath, nG, nH);

					insertNodePath(openListPath, nStancePath, max_number_paths);
				}
			}

			if (openListPath.size() == 0)
			{
				flag_continue = false;
			}
		}

		return retPath;
	}

	void insertNodePath(std::list<mStancePathNode>& openList, mStancePathNode& nNode, int max_num)
	{
		int k = 0;
		std::list<mStancePathNode>::iterator iter_pathNode = openList.begin();
		while (iter_pathNode != openList.end())
		{
			mStancePathNode& n_i = *iter_pathNode;//openList[i];
			float cF = n_i.mG + n_i.mH;
			float nF = nNode.mG + nNode.mH;
			if (nF < cF)
			{
				break;
			}
			iter_pathNode++;
			k++;
		}
		if (k < max_num)
			openList.insert(iter_pathNode, nNode);

		if ((int)openList.size() > max_num)
		{
			openList.pop_back();
		}
		return;
	}
	//////////////////////////////////////////////////////////////

	std::vector<int> reversePath(std::vector<int>& _iPath)
	{
		std::vector<int> nPath;
		for (int i = _iPath.size() - 1; i >=0; i--)
		{
			nPath.push_back(_iPath[i]);
		}
		return nPath;
	}

	//////////////////////////////////////////////////////// the huristic values to goal is not admisible
	float getCostToGoal(std::vector<int>& iCStance)
	{
		/*float cCost = 0.0f; 
		for (unsigned int i = 0; i < iCStance.size(); i++)
		{ 
			if (iCStance[i] != -1)
			{
				cCost += (mClimberSampler->getHoldPos(iCStance[i]) - goalPos).squaredNorm();
			}
			else
			{
				cCost += 0.0f;
			}
		}
		return 0.01f * sqrt(cCost);*/
		float cCount = 0.0f;
		std::vector<Vector3> sample_desired_hold_points;
		Vector3 midPoint = mClimberSampler->getHoldStancePosFrom(iCStance, sample_desired_hold_points, cCount);
		return (midPoint - goalPos).norm();
	}
	public:
	float getCostAtNode(std::vector<int>& iCStance, bool printDebug=false)
	{
		float k_crossing = 100;
		float k_hanging_hand = 200 + 50; // 200
		float k_hanging_leg = 10 + 10; // 10
//		float k_hanging_more_than2 = 0;//100;
		float k_matching = 100;
		float k_dis = 1000;

		float _cost = 0.0f;

		// punish for hanging more than one limb
		int counter_hanging = 0;
		for (unsigned int i = 0; i < iCStance.size(); i++)
		{
			if (iCStance[i] == -1)
			{
				counter_hanging++;
				
				if (i >= 2)
				{
					// punish for having hanging hand
					_cost += k_hanging_hand;
					if (printDebug) rcPrintString("Hanging hand!");
				}
				else
				{
					// punish for having hanging hand
					_cost += k_hanging_leg;
				}
			}
		}

		//// punish for having two or more limbs hanging
		//if (counter_hanging >= 2) 
		//{
		//	_cost += k_hanging_more_than2;
		//	if (printDebug) rcPrintString("More than 2 hanging limbs!");
		//}

		// crossing hands
		if (iCStance[2] != -1 && iCStance[3] != -1)
		{
			Vector3 rHand = mClimberSampler->getHoldPos(iCStance[3]);
			Vector3 lHand = mClimberSampler->getHoldPos(iCStance[2]);

			if (rHand.x() < lHand.x())
			{
				_cost += k_crossing;
				if (printDebug) rcPrintString("Hands crossed!");
			}
		}

		// crossing feet
		if (iCStance[0] != -1 && iCStance[1] != -1)
		{
			Vector3 lLeg = mClimberSampler->getHoldPos(iCStance[0]);
			Vector3 rLeg = mClimberSampler->getHoldPos(iCStance[1]);

			if (rLeg.x() < lLeg.x())
			{
				_cost += k_crossing;
				if (printDebug) rcPrintString("Legs crossed!");
			}
		}

		// crossing hand and foot
		for (unsigned int i = 0; i <= 1; i++)
		{
			if (iCStance[i] != -1)
			{
				Vector3 leg = mClimberSampler->getHoldPos(iCStance[i]);
				for (unsigned int j = 2; j <= 3; j++)
				{
					if (iCStance[j] != -1)
					{
						Vector3 hand = mClimberSampler->getHoldPos(iCStance[j]);
						
						if (hand.z() <= leg.z())
						{
							_cost += k_crossing;
						}
					}
				}
			}
		}

		//feet matching
		if (iCStance[0]==iCStance[1])
		{
			_cost += k_matching;
			if (printDebug) rcPrintString("Feet matched!");
		}

		//punishment for hand and leg being close
		for (unsigned int i = 0; i <= 1; i++)
		{
			if (iCStance[i] != -1)
			{
				Vector3 leg = mClimberSampler->getHoldPos(iCStance[i]);
				for (unsigned int j = 2; j <= 3; j++)
				{
					if (iCStance[j] != -1)
					{
						Vector3 hand = mClimberSampler->getHoldPos(iCStance[j]);
						
						float cDis = (hand - leg).norm();
						
						const float handAndLegDistanceThreshold = 0.5f;//mClimberSampler->climberRadius / 2.0f;
						if (cDis < handAndLegDistanceThreshold)
						{
							cDis/=handAndLegDistanceThreshold;
							_cost += k_dis*std::max(0.0f,1.0f-cDis);
							if (printDebug) rcPrintString("Hand and leg too close! v = %f", k_dis*std::max(0.0f,1.0f-cDis));
						}
					}
				}
			}
		}

		return _cost;
	}
	private:
	std::vector<Vector3> getExpectedPositionSigma(Vector3 midPoint)
	{
		float r = mClimberSampler->climberRadius;
		
		std::vector<float> theta_s;
		theta_s.push_back(PI + PI / 4.0f);
		theta_s.push_back(1.5 * PI + PI / 4.0f);
		theta_s.push_back(0.5 * PI + PI / 4.0f);
		theta_s.push_back(PI / 4.0f);

		std::vector<Vector3> expectedPoses;
		for (unsigned int i = 0; i < theta_s.size(); i++)
		{
			Vector3 iDir(cosf(theta_s[i]), 0.0f, sinf(theta_s[i]));
			expectedPoses.push_back(midPoint + (r / 2.0f) * iDir);
		}

		return expectedPoses;
	}

	float getDisFromStanceToStance(std::vector<int>& si, std::vector<int>& sj)
	{
		float cCount = 0.0f;
		std::vector<Vector3> hold_points_i;
		Vector3 midPoint1 = mClimberSampler->getHoldStancePosFrom(si, hold_points_i, cCount);
//		std::vector<Vector3> e_hold_points_i = getExpectedPositionSigma(midPoint1);

		std::vector<Vector3> hold_points_j;
		Vector3 midPoint2 = mClimberSampler->getHoldStancePosFrom(sj, hold_points_j, cCount);
//		std::vector<Vector3> e_hold_points_j = getExpectedPositionSigma(midPoint2);

		float cCost = 0.0f;
		float hangingLimbExpectedMovement=2.0f;
		for (unsigned int i = 0; i < si.size(); i++)
		{
			float coeff_cost = 1.0f;
			if (si[i] != sj[i])
			{
				Vector3 pos_i;
				if (si[i] != -1)
				{
					pos_i = hold_points_i[i];
				}
				else
				{
					//pos_i = e_hold_points_i[i];
					cCost += 0.5f;
					continue;
				}
				Vector3 pos_j;
				if (sj[i] != -1)
				{
					pos_j = hold_points_j[i];
				}
				else
				{
					//pos_j = e_hold_points_j[i];
					cCost += hangingLimbExpectedMovement;
					continue;
				}

				//favor moving hands
				if (i >= 2)
					coeff_cost = 0.9f;

				cCost += coeff_cost * (pos_i - hold_points_j[i]).squaredNorm();
			}
			else
			{
				if (sj[i] == -1)
				{
					cCost += hangingLimbExpectedMovement;
				}
			}
		}

		return sqrtf(cCost);
	}
	
	bool firstHoldIsLower(int hold1, int hold2)
	{
		if (hold1==-1 && hold2==-1)
			return false;
		if (hold1!=-1 && mClimberSampler->getHoldPos(hold1).z() < mClimberSampler->getHoldPos(hold2).z())
		{
			return true;
		}
		//first hold is "free" => we can't really know 
		return false;
	}

	float getCostMovementLimbs(std::vector<int>& si, std::vector<int>& sj)
	{
		float k_dis = 1.0f;
		float k_2limbs = 120.0f;//100.0f;
//		float k_freeAnother = 0.0f;//20.0f;
		float k_pivoting_close_dis = 500.0f;

		//First get the actual distance between holds. We scale it up 
		//as other penalties are not expressed in meters
		float _cost = k_dis * getDisFromStanceToStance(si, sj);

		//penalize moving 2 limbs, except in "ladder climbing", i.e., moving opposite hand and leg
		bool flag_punish_2Limbs = true;
		bool is2LimbsPunished = false;
		if (mNode::getDiffBtwSetASetB(si, sj) > 1.0f)
		{

			if (si[0] != sj[0] && si[3] != sj[3] && firstHoldIsLower(si[0],sj[0]))
			{
				flag_punish_2Limbs = false;
				if (sj[0] != -1 && sj[3] != -1 && mClimberSampler->getHoldPos(sj[3]).x() - mClimberSampler->getHoldPos(sj[0]).x() < 0.5f)
					flag_punish_2Limbs = true;
				if (sj[0] != -1 && sj[3] != -1 && mClimberSampler->getHoldPos(sj[3]).z() - mClimberSampler->getHoldPos(sj[0]).z() < 0.5f)
					flag_punish_2Limbs = true;
			}

			if (si[1] != sj[1] && si[2] != sj[2] && firstHoldIsLower(si[1],sj[1]))
			{
				flag_punish_2Limbs = false;
				if (sj[1] != -1 && sj[2] != -1 && mClimberSampler->getHoldPos(sj[1]).x() - mClimberSampler->getHoldPos(sj[2]).x() < 0.5f)
					flag_punish_2Limbs = true;
				if (sj[1] != -1 && sj[2] != -1 && mClimberSampler->getHoldPos(sj[2]).z() - mClimberSampler->getHoldPos(sj[1]).z() < 0.5f)
					flag_punish_2Limbs = true;
			}

			//if (si[0] == -1 && si[1] == -1)
			//{
			//	int notifyme = 1;
			//}
			if (flag_punish_2Limbs)
				_cost += k_2limbs;
		}
		/*else
		{
			if ((si[0] == -1 || si[1] == -1) && (si[2] == 2 && si[3] == 3))
			{
				int notifyme = 1;
			}
		}*/

		////penalize transitions where we will have both legs momentarily hanging 
		//if ((si[0]==-1 && (si[1] !=sj[1])) || (si[1]==-1 && (si[0] !=sj[0])))
		//{
		//	_cost+=150.0f;
		//}


		//One of the hand on a foot hold and the other hand is moved.
		//Commented out, as we are already punishing nodes where hand and foot are on the same hold
		//if (si[2] != sj[2] || si[3] != sj[3])
		//{
		//	if (si[2] != sj[2])
		//	{
		//		if ((si[3] == si[0] || si[3] == si[1]) && si[3] >= 0)
		//		{
		//			_cost += 100;
		//		}
		//	}
		//	if (si[3] != sj[3])
		//	{
		//		if ((si[2] == si[0] || si[2] == si[1]) && si[2] >= 0)
		//		{
		//			_cost += 100;
		//		}
		//	}
		//}

		// calculating the stance during the transition
		std::vector<int> sn(4);
		int count_free_limbs = 0;
		for (unsigned int i = 0; i < si.size(); i++)
		{
			if (si[i] != sj[i])
			{
				sn[i]=-1;
				count_free_limbs++;
			}
			else
			{
				sn[i] = si[i];
			}
		}
		// free another
		if (count_free_limbs >= 2 && mNode::getDiffBtwSetASetB(si, sj) == 1.0f)
			_cost += k_2limbs;

		// punish for pivoting!!!
		float v = 0.0f;
		float max_dis = -FLT_MAX;
		for (unsigned int i = 0; i <= 1; i++)
		{
			if (sn[i] != -1)
			{
				Vector3 leg = mClimberSampler->getHoldPos(sn[i]);
				for (unsigned int j = 2; j <= 3; j++)
				{
					if (sn[j] != -1)
					{
						Vector3 hand = mClimberSampler->getHoldPos(sn[j]);

						float cDis = (hand - leg).norm();

						if (max_dis < cDis)
							max_dis = cDis;
					}
				}
			}
		}
		if (max_dis >= 0 && max_dis < mClimberSampler->climberRadius / 2.0f && count_free_limbs > 1.0f)
		{
			v += k_pivoting_close_dis;
		}
		_cost += v;

		return _cost;
	}

	float getCostTransition(std::vector<int>& si, std::vector<int>& sj)
	{
		return 1.0f;
	}

	////////////////////////////////////////////////////////
	std::vector<int> returnPathFrom(int iStanceIndex)
	{
		std::vector<int> rPath;

		int cIndex = iStanceIndex;

		int counter = 0;
		while (cIndex >= 0) // cIndex == 0 is (-1,-1,-1,-1)
		{
			mStanceNode nSi = stance_nodes[cIndex];
			rPath.push_back(nSi.stanceIndex);

			cIndex = nSi.bFatherStanceIndex;

			counter++;

			if (counter > (int)stance_nodes.size())
				break;
		}

		return rPath;
	}

	std::vector<std::vector<int>> returnPathFrom(std::vector<int>& pathIndex)
	{
		std::vector<std::vector<int>> lPath;
		for (int i = pathIndex.size() - 1; i >= 0; i--)
		{
			mStanceNode nSi = stance_nodes[pathIndex[i]];
			lPath.push_back(nSi.hold_ids);
		}
		return lPath;
	}

	bool isGoalFound(int iStanceIndex)
	{
		return mSampler::isInSetHoldIDs(iStanceIndex, goal_stances);
	}

	void setTriedToCurrentFather(int cChildIndex, int cFather, bool val)
	{
		if (cFather < 0)
			return;
		for (unsigned int i = 0; i < stance_nodes[cChildIndex].parentStanceIds.size(); i++)
		{
			if (stance_nodes[cChildIndex].parentStanceIds[i] == cFather)
			{
				stance_nodes[cChildIndex].isItTried_to_father[i] = val;
				return;
			}
		}
		return;
	}

	void setTriedToCurrentChild(int cFatherIndex, int cChild, bool val)
	{
		if (cFatherIndex < 0 || cChild < 0)
			return;
		for (unsigned int i = 0; i < stance_nodes[cFatherIndex].childStanceIds.size(); i++)
		{
			if (stance_nodes[cFatherIndex].childStanceIds[i] == cChild)
			{
				stance_nodes[cFatherIndex].isItTried_to_child[i] = val;
				return;
			}
		}
		return;
	}

	bool isItTriedToCurrentFather(int cChildIndex, int cFather)
	{
		for (unsigned int i = 0; i < stance_nodes[cChildIndex].parentStanceIds.size(); i++)
		{
			if (stance_nodes[cChildIndex].parentStanceIds[i] == cFather)
			{
				return stance_nodes[cChildIndex].isItTried_to_father[i];
			}
		}
		return false;
	}
	
	bool isAllOfNodeChildrenTried(int cNodeIndex)
	{
		for (unsigned int i = 0; i < stance_nodes[cNodeIndex].isItTried_to_child.size(); i++)
		{
			if (!stance_nodes[cNodeIndex].isItTried_to_child[i])
			{
				return false;
			}
		}
		return true;
	}

	bool isLoopCreated(int iStanceIndex)
	{
		int cIndex = iStanceIndex;

		int counter = 0;
		while (cIndex != 0)
		{
			mStanceNode nSi = stance_nodes[cIndex];

			cIndex = nSi.bFatherStanceIndex;

			counter++;

			if (counter > (int)stance_nodes.size())
			{
				printf("Loooooooooooooooooooooooooooooop");
				return true;
			}
		}
		return false;
	}

	int addGraphNode(int _fromIndex, std::vector<int>& _sStance)
	{
		AALTO_ASSERT1(_sStance.size()==4);
		mStanceNode nStance(_sStance);
		nStance.stanceIndex = stance_nodes.size();

		if (_fromIndex == -1)
		{
			nStance.nodeCost = 0.0f;

			stance_nodes.push_back(nStance);
			int index = stance_nodes.size() - 1;
			stanceToNode[stanceToKey(nStance.hold_ids)]=index;
			return index;
		}

		int stance_id = findStanceFrom(nStance.hold_ids);

		if (stance_id == -1)
		{
			nStance.nodeCost = getCostAtNode(_sStance);

			if (nStance.hold_ids[3] != -1) // search for right hand
			{
				//printf("%d \n", nStance.hold_ids[3]);
				Vector3 holdPos = mClimberSampler->getHoldPos(nStance.hold_ids[3]);
				if ((holdPos - goalPos).norm() < 0.1f)
				{
					mSampler::addToSetHoldIDs(nStance.stanceIndex, goal_stances);
				}
			}
			if (nStance.hold_ids[2] != -1) // search for right hand
			{
				//printf("%d \n", nStance.hold_ids[2]);
				Vector3 holdPos = mClimberSampler->getHoldPos(nStance.hold_ids[2]);
				if ((holdPos - goalPos).norm() < 0.1f)
				{
					mSampler::addToSetHoldIDs(nStance.stanceIndex, goal_stances);
				}
			}

			stance_nodes.push_back(nStance);
			if (stance_nodes.size() % 100 ==0)  //don't printf every node, will slow things down
				printf("Number of nodes: %d\n",stance_nodes.size());
			int index = stance_nodes.size() - 1;
			stance_id = index;
			stanceToNode[stanceToKey(nStance.hold_ids)]=index;
		}
		else
		{
			//if (stance_nodes[stance_id].stanceIndex == 1575)
			//{
			//	int notifyme= 1;
			//}
			//if (stance_nodes[stance_id].stanceIndex != stance_id)
			//{
			//	int notifyme = 1;
			//}
			if (stance_nodes[stance_id].stanceIndex == _fromIndex)
			{
				return stance_nodes[stance_id].stanceIndex;
			}
		}

		if (mSampler::addToSetHoldIDs(stance_nodes[_fromIndex].stanceIndex, stance_nodes[stance_id].parentStanceIds))
		{
			stance_nodes[stance_id].isItTried_to_father.push_back(false);
			if (stance_nodes[stance_id].parentStanceIds.size() > maxGraphDegree)
				maxGraphDegree = stance_nodes[stance_id].parentStanceIds.size();
		}

		if (mSampler::addToSetHoldIDs(stance_nodes[stance_id].stanceIndex, stance_nodes[_fromIndex].childStanceIds))
		{
			stance_nodes[_fromIndex].cost_moveLimb_to_child.push_back(getCostMovementLimbs(stance_nodes[_fromIndex].hold_ids, stance_nodes[stance_id].hold_ids));
			stance_nodes[_fromIndex].cost_transition_to_child.push_back(getCostTransition(stance_nodes[_fromIndex].hold_ids, stance_nodes[stance_id].hold_ids));
			stance_nodes[_fromIndex].isItTried_to_child.push_back(false);

			if (stance_nodes[_fromIndex].childStanceIds.size() > maxGraphDegree)
				maxGraphDegree = stance_nodes[_fromIndex].childStanceIds.size();
		}

		return stance_id;
	}

	int mRootGraph;

	std::vector<int> goal_stances;
	std::vector<mStanceNode> stance_nodes;
	unsigned int maxGraphDegree;
};

class mTrialStructure
{
public:
	mTrialStructure(int iTreeNode, int iGraPhNode, int iGraphFatherNodeId, float iValue)
	{
		_treeNodeId = iTreeNode;
		_graphNodeId = iGraPhNode;
		_graphFatherNodeId = iGraphFatherNodeId;
		val = iValue;
	}

	int _treeNodeId;
	int _graphNodeId;
	int _graphFatherNodeId;
	float val;
};

class mRRT
{
public:
	mSampler* mClimberSampler;

	mStanceGraph mySamplingGraph;

	std::vector<mNode> mRRTNodes; // each node has the state of the agent
	KDTree<int> mKDTreeNodeIndices; // create an index on mRRTNodes
	int indexRootTree;

	mController* mOptimizer;
	SimulationContext* mContextRRT;
	mSampleStructure mSample;
	
	int itr_optimization_for_each_sample;
	int max_itr_optimization_for_each_sample;
	int max_waste_interations;
	int max_no_improvement_iterations;

	int total_samples_num;
	int accepted_samples_num;

	// variable for playing animation
	BipedState lOptimizationBipedState;
	std::vector<int> path_nodes_indices;
	bool isPathFound;
	int lastPlayingNode;
	int lastPlayingStateInNode;
	bool isNodeShown;
	double cTimeElapsed;
	bool isAnimationFinished;

	// handling initial stance
	std::vector<int> initial_stance;
	bool initialstance_exists_in_tree;
	int index_node_initial_stance;

	// hanging leg when they are on a same hold
	bool isReachedLegs[2];

	// growing tree given a path by A*
	std::vector<mTrialStructure> mPriorityQueue;
	std::vector<std::vector<int>> mNodeIDsForPath;
	std::vector<bool> isItReachedFromToOnPath;
	std::vector<std::vector<int>> desiredPath;
	std::vector<int> mTriedPathIndicies;
	float cCostAStarPath;

	//handling offline - online method
	enum targetMethod{Offiline = 0, Online = 1};

	// handling reached goals
	std::vector<int> goal_nodes;
	int cGoalPathIndex;
	bool isGoalReached;
	float cCostGoalMoveLimb;
	float cCostGoalControlCost;

	// handling rewiring
	int numRefine; // for debug
	int numPossRefine; // for debug
	std::vector<Vector2> mTransitionFromTo;

	// variables for sampling
	std::vector<int> m_reached_hold_ids; // reached to holds IDs
	std::vector<int> m_future_reachable_hold_ids; // future reachable holds are used to sample hand pos based on the calculated gain

	mRRT(SimulationContext* iContextRRT, mController* iController, mSampler* iHoldSampler)
		:mKDTreeNodeIndices((int)(iController->startState.bodyStates.size() * 3))
	{
		mOptimizer = iController;
		mContextRRT = iContextRRT;
		mClimberSampler = iHoldSampler;

		// handling initial stance
		initial_stance.push_back(-1);
		initial_stance.push_back(-1);
		initial_stance.push_back(2);
		initial_stance.push_back(3);
		initialstance_exists_in_tree = false;
		index_node_initial_stance = -1;

		mySamplingGraph.mClimberSampler = mClimberSampler;
		mySamplingGraph.goalPos = mContextRRT->getGoalPos();
		mySamplingGraph.buildGraph(mOptimizer->startState.hold_bodies_ids, initial_stance);

		// handling reached goal
		isGoalReached = false;
		cCostGoalControlCost = -1;
		cCostGoalMoveLimb = -1;
		cGoalPathIndex = 0;
		numRefine = 0;
		numPossRefine = 0;

		isReachedLegs[0] = false;
		isReachedLegs[1] = false;
		UpdateTreeForGraph(mOptimizer->startState, -1, std::vector<BipedState>(), mSampleStructure()); // -1 means root
		indexRootTree = 0;

		max_waste_interations = useOfflinePlanning ? 0 : 5;//5 if online

		//the online method is not working good with 3*(int)(cTime/nPhysicsPerStep), its value should be around 50
		max_itr_optimization_for_each_sample = useOfflinePlanning ? 3*(int)(cTime/nPhysicsPerStep) : (int)(1.5f * cTime);
		itr_optimization_for_each_sample = 0;
		//the online method is not working good with (int)(cTime/nPhysicsPerStep)/4, its value should be around 8
		max_no_improvement_iterations = useOfflinePlanning ? 5 : (int)(cTime/4);

		total_samples_num = 0;
		accepted_samples_num = 0;

		// variable for playing animation
		lOptimizationBipedState = mOptimizer->startState.getNewCopy(mContextRRT->getNextFreeSavingSlot(), mContextRRT->getMasterContextID());
		isPathFound = false;
		lastPlayingNode = 0;
		lastPlayingStateInNode = 0;
		isNodeShown = false;
		cTimeElapsed = 0.0f;
		isAnimationFinished = false;

		cCostAStarPath = 0.0f;
	} 

	void mRunPathPlanner(targetMethod itargetMethd, bool advance_time, bool flag_play_animation, mCaseStudy icurrent_case_study)
	{
		if (itargetMethd == targetMethod::Offiline)
		{
			mRunPathPlannerOffline(flag_play_animation, advance_time, icurrent_case_study);
		}
		else
		{
			mRunPathPlannerOnline(advance_time, flag_play_animation, icurrent_case_study);
		}
	}

	void mPrintStuff()
	{
		rcPrintString("Cost of A* prune stance path: %f \n",cCostAStarPath);
		//compute destination stance cost just to print out the components
		//mySamplingGraph.getCostAtNode(mSample.desired_hold_ids,true);

		/*printf("\n current: %d, %d, %d, %d, start = %d, %d, %d, %d, desreid= %d, %d, %d, %d, TO: %d,%d,%d,%d \n", mController->startState.hold_bodies_ids[0]
							, mController->startState.hold_bodies_ids[1], mController->startState.hold_bodies_ids[2], mController->startState.hold_bodies_ids[3]
							,mSample.initial_hold_ids[0], mSample.initial_hold_ids[1], mSample.initial_hold_ids[2], mSample.initial_hold_ids[3]
							,mSample.desired_hold_ids[0], mSample.desired_hold_ids[1], mSample.desired_hold_ids[2], mSample.desired_hold_ids[3]
							,mSample.to_hold_ids[0], mSample.to_hold_ids[1], mSample.to_hold_ids[2], mSample.to_hold_ids[3]);

		if (mSample.toNode != -1)
		{
			printf("\n //////////////  Error: %f ////////////////// \n", mRRTNodes[mSample.toNode].getDisFrom(mController->startState));
		}
		else
		{
			printf("\n //////////////  Error: %f ////////////////// \n", -1.0f);
		}

		printf("\n num nodes: %d,from node: %d, cCostGoal:%f, NRef:%d, NPRef:%d \n", mRRTNodes.size(), mSample.closest_node_index, cCostGoal, numRefine, numPossRefine);*/
	}

private:
	/////////////////////////////////////////////// choose offline or online mode
	void mRunPathPlannerOffline(bool flag_play_animation, bool advance_time, mCaseStudy icurrent_case_study)
	{
		if (!flag_play_animation)
		{
			// reverting to last state before animation playing
			if (mSample.restartSampleStartState && mSample.isSet)
			{
				//mController->startState = lOptimizationBipedState; // this is for online mode
				mOptimizer->startState = mRRTNodes[mSample.closest_node_index].cNodeInfo;
			}

			// get another sigma to simulate forward
			if (!mSample.isSet || mSample.isReached || mSample.isRejected || itr_optimization_for_each_sample < max_waste_interations)
			{
				if (!mSample.isSet || mSample.isReached || mSample.isRejected)
					mSample = mRRTTowardSamplePath2(mSample); 
				mSample.statesFromTo.clear();

				//initiate before simulation
				if (mSample.closest_node_index != -1)
				{
					mOptimizer->startState = mRRTNodes[mSample.closest_node_index].cNodeInfo;
				}
				// means to restart to initial setting for holds
				if (mSample.isSet)
				{
					setArmsLegsHoldPoses();
				}
			}

			if (mSample.isSet)
			{
//				rcPrintString("Low-level controller iteration %d",itr_optimization_for_each_sample);
				// when it is false, it mean offline optimization
				mSteerFunc(false);
				if (advance_time)
					itr_optimization_for_each_sample++;

				// offline optimization is done, simulate forward on the best trajectory
				if ((advance_time && mSample.numItrFixedCost > max_no_improvement_iterations)
					|| itr_optimization_for_each_sample > max_itr_optimization_for_each_sample)
				{
					itr_optimization_for_each_sample = 0;
					bool flag_simulation_break = false;
					int maxTimeSteps=nTimeSteps/nPhysicsPerStep;
					/*if (optimizerType==otCMAES)
					{
						maxTimeSteps=mController->bestCmaesTrajectory.nSteps;
						if (nPhysicsPerStep!=1)
							Debug::throwError("CMAES does not support other than nPhysicsPerStep==1");
					}*/
					for (int cTimeStep = 0; cTimeStep < maxTimeSteps && !flag_simulation_break; cTimeStep++)
					{
						mStepOptimization(cTimeStep);
						// connect contact pos i to the desired hold pos i if some condition (for now just distance) is met
						m_Connect_Disconnect_ContactPoint(mSample.desired_hold_ids);

						// if the simulation is rejected a new sample should be generated
						// if the simulation is accepted the control sequence should be restored and then new sample should be generated
						mAcceptOrRejectSample(mSample, targetMethod::Offiline);

						if (mSample.isRejected)
							flag_simulation_break = true;
					}
					if (!flag_simulation_break)
					{
						// we just set the termination condition true to add new node in offline
						itr_optimization_for_each_sample = max_itr_optimization_for_each_sample + 1;

						mAcceptOrRejectSample(mSample, targetMethod::Offiline);
						mSample.isReached = true;
					}
				}

				mPrintStuff();
			}
			else
			{
				mOptimizer->syncMasterContextWithStartState(true);
			}
			if (mSample.closest_node_index >= 0)
				mSample.draw_ws_points(mRRTNodes[mSample.closest_node_index].cNodeInfo.getEndPointPosBones(SimulationContext::BodyName::BodyTrunk));
			
//			// save last optimization state
//			lOptimizationBipedState = mController->startState.getNewCopy(lOptimizationBipedState.saving_slot_state, mContextRRT->getMasterContextID());//
			isPathFound = false;
		}
		else
		{
			playAnimation(icurrent_case_study);
		}
		return;
	}

	void mRunPathPlannerOnline(bool advance_time, bool flag_play_animation, mCaseStudy icurrent_case_study)
	{
		if (!flag_play_animation)
		{
			if (mSample.restartSampleStartState && mSample.isSet)
			{
				mOptimizer->startState = lOptimizationBipedState;
			}
			if (!mSample.isSet || mSample.isReached || mSample.isRejected || itr_optimization_for_each_sample < max_waste_interations)
			{
				if (!mSample.isSet || mSample.isReached || mSample.isRejected)
					mSample = mRRTTowardSamplePath2(mSample); //mRRTTowardSample(); 
				mSample.statesFromTo.clear();
				if (mSample.closest_node_index != -1)
				{
					mOptimizer->startState = mRRTNodes[mSample.closest_node_index].cNodeInfo;
				}
				// means to restart to initial setting for holds
				if (mSample.isSet)
				{
					setArmsLegsHoldPoses();
				}
			}

			if (mSample.isSet)
			{
				mSteerFunc(advance_time);

				if (advance_time)
				{
					// connect contact pos i to the desired hold pos i if some condition (for now just distance) is met
					m_Connect_Disconnect_ContactPoint(mSample.desired_hold_ids);

					// if the simulation is rejected a new sample should be generated
					// if the simulation is accepted the control sequence should be restored and then new sample should be generated
					mAcceptOrRejectSample(mSample, targetMethod::Online);

					itr_optimization_for_each_sample++;
				}
				mPrintStuff();
			}
			else
			{
				mOptimizer->syncMasterContextWithStartState(true);
			}
			if (mSample.closest_node_index >= 0)
				mSample.draw_ws_points(mRRTNodes[mSample.closest_node_index].cNodeInfo.getEndPointPosBones(SimulationContext::BodyName::BodyTrunk));
			
			// save last optimization state
			lOptimizationBipedState = mOptimizer->startState.getNewCopy(lOptimizationBipedState.saving_slot_state, mContextRRT->getMasterContextID());
			isPathFound = false;
		}
		else
		{
			playAnimation(icurrent_case_study);
		}
		return;
	}

	/////////////////////////////// my method /////////////////////////////////////////////////////////////
	mSampleStructure mRRTTowardSamplePath2(mSampleStructure& pSample)
	{		
		updatePriorityQueueReturnSample2(pSample);

		if (mPriorityQueue.size() == 0)
		{
			updateGraphTransitionValues2();

			mySamplingGraph.solveGraph(cCostAStarPath, mStanceGraph::mAlgSolveGraph::myDijkstraHueristic);

			desiredPath = mySamplingGraph.returnPath();

			getAllNodesAroundThePath2(desiredPath);

			mTriedPathIndicies.clear();
		}

		if (mPriorityQueue.size() > 0)
		{
			mTrialStructure _tryFromTo = mPriorityQueue[0];
			mPriorityQueue.erase(mPriorityQueue.begin());

			return retSampleFromGraph(_tryFromTo._graphFatherNodeId, _tryFromTo._graphNodeId);
		}
		return mSampleStructure();
	}

	void updatePriorityQueueReturnSample2(mSampleStructure& pSample)
	{
		if (pSample.isSet)
		{
			if (isItReached(pSample))
			{
				int index_on_path = mTriedPathIndicies[mTriedPathIndicies.size() - 1];
				isItReachedFromToOnPath[index_on_path - 1] = true;
				std::vector<int> mNodeIDsForPath_i;
				mNodeIDsForPath_i.push_back(pSample.toNode);
					//getAllTreeNodesForPathIndex2(desiredPath, index_on_path);

				if (mNodeIDsForPath_i.size() > 0)
				{
					mNodeIDsForPath[index_on_path] = mNodeIDsForPath_i;
					if (index_on_path < (int)isItReachedFromToOnPath.size())
					{
						if (!mSampler::isInSetHoldIDs(index_on_path + 1, mTriedPathIndicies)) //!isItReachedFromToOnPath[index_on_path] && 
						{
							mTrialStructure nT(-1, index_on_path+1, index_on_path, -1);
							if (!isTrialStrExistsInQueue(nT))
								mPriorityQueue.insert(mPriorityQueue.begin(), nT);
						}
					}
				}
			}
			else
			{
				int index_on_path = mTriedPathIndicies[mTriedPathIndicies.size() - 1];
				isItReachedFromToOnPath[index_on_path - 1] = false;
			}
		}

		if (mPriorityQueue.size() == 0 && mTriedPathIndicies.size() == 0)
		{
			for (unsigned int i = 0; i < isItReachedFromToOnPath.size(); i++)
			{
				if (!isItReachedFromToOnPath[i] && mNodeIDsForPath[i].size() > 0)
				{
					mTrialStructure nT(-1, i+1, i, -1);
					if (!isTrialStrExistsInQueue(nT))
						mPriorityQueue.push_back(nT);
				}
			}
		}

		return;
	}

	void getAllNodesAroundThePath2(std::vector<std::vector<int>>& dPath)
	{
		mNodeIDsForPath.clear();
		isItReachedFromToOnPath.clear();

		std::vector<std::vector<int>> mNodeIDsForStancePath;

		for (int i = 0; i < (int)dPath.size() - 1; i++)
		{
			isItReachedFromToOnPath.push_back(false);
		}

		// nodes that are representing stance path
		for (int i = 0; i < (int)dPath.size(); i++)
		{
			std::vector<int> mNodeIDsForStancePath_i;
			std::vector<int> mNodeIDsForPath_i = getAllTreeNodesForPathIndex2(dPath, i, mNodeIDsForStancePath_i);

			mNodeIDsForStancePath.push_back(mNodeIDsForStancePath_i);
			mNodeIDsForPath.push_back(mNodeIDsForPath_i);
		}
		//Uncomment to get all the nodes instead of just the one that follows the stance path
		//for (int i = 0; i < (int)dPath.size(); i++)
		//{
		//	if (mNodeIDsForPath[i].size() == 0)
		//	{
		//		mNodeIDsForPath[i] = mNodeIDsForStancePath[i];
		//	}
		//}
		return;
	}

	std::vector<int> getAllTreeNodesForPathIndex2(std::vector<std::vector<int>>& dPath, int i
		, std::vector<int>& mNodeIDsForStancePath_i)
	{
		float cCost = 0.0f;
		std::vector<int> s_i; 
		std::vector<int> s_i_1;

		std::vector<int> mNodeIDsForPath_i;
		if (i == 0)
		{
			mNodeIDsForPath_i.push_back(0);
			return mNodeIDsForPath_i;
		}

		for (int j = 0; j < (int)dPath.size(); j++)
		{
			s_i = dPath[j];
			if (j-1 >= 0)
			{
				s_i_1 = dPath[j-1];
				cCost += mNode::getDiffBtwSetASetB(s_i_1, s_i);
			}
			if (j == i)
			{
				break;
			}
		}

		// now we find all nodes that represent stance i in dPath and they should have come from i - 1
		std::vector<int> s_i_p1;
		if (i+1 < (int)dPath.size())
		{
			s_i_p1 = dPath[i+1];
		}

		for (unsigned int n = 0; n < mRRTNodes.size(); n++)
		{
			mNode* tNode_n = &mRRTNodes[n];

			if (!tNode_n->isNodeEqualTo(s_i))
			{
				continue;
			}

			if (s_i_p1.size() > 0)
			{
				if (tNode_n->isInTriedHoldSet(s_i_p1) && !isStanceExistInTreeNodeChildren(tNode_n, s_i_p1))
				{
					continue;
				}
			}

			if (i - 1 >= 0)
			{
				if (mSampler::isInSetHoldIDs(tNode_n->mFatherIndex, mNodeIDsForPath[i-1]))
				{
					isItReachedFromToOnPath[i-1] = true;
					mNodeIDsForPath_i.push_back(n); // representing s_i in the path
				}
			}
			else if (tNode_n->mFatherIndex == -1)
			{
				mNodeIDsForPath_i.push_back(n);
			}
			if (!isStanceExistInTreeNodeChildren(tNode_n, s_i_p1))
			{
				mNodeIDsForStancePath_i.push_back(n);
			}
			//if (s_i_1.size() > 0)
			//{
			//	bool flag_add = false;
			//	/*if (getNodeCost(tNode_n) <= cCost)
			//	{
			//		flag_add = true;
			//	}*/
			//	if (tNode_n->mFatherIndex == -1)
			//	{
			//		if (!flag_add)
			//			continue;
			//	}
			//	mNode* tNode_f = &mRRTNodes[tNode_n->mFatherIndex];
			//	if (!tNode_f->isNodeEqualTo(s_i_1))
			//	{
			//		if (!flag_add)
			//			continue;
			//	}
			//	else
			//	{
			//		isItReachedFromToOnPath[i-1] = true;
			//	}
			//}
		}
		return mNodeIDsForPath_i;
	}

	void updateGraphTransitionValues2()
	{
		bool flag_path_changed = false;
		for (unsigned int i = 0; i < mTriedPathIndicies.size(); i++)
		{
			int index_on_path = mTriedPathIndicies[i];
			if (!isItReachedFromToOnPath[index_on_path - 1])
			{
				bool is_changed_val = mySamplingGraph.updateGraphNN(mySamplingGraph.getIndexStanceNode(index_on_path-1), mySamplingGraph.getIndexStanceNode(index_on_path));
				if (!flag_path_changed)
				{
					flag_path_changed = is_changed_val;
				}
			}
		}

		if (flag_path_changed)
		{
			mySamplingGraph.initializeOpenListAStarPrune();
		}

		if (!flag_path_changed && mySamplingGraph.retPath.size() > 0) // path is reached, A* should return another path
		{
			if (!mSampler::isInSampledStanceSet(mySamplingGraph.retPath, mySamplingGraph.m_found_paths))
			{
				mySamplingGraph.m_found_paths.push_back(mySamplingGraph.retPath);
			}
		}

		return;
	}

	/////////////////////////////// utilities for samplings form graph /////////////////////////////////////
	template<typename T>
	void createParticleList(unsigned int mN, float cWeight, T cSample,std::vector<float>& weights, float& sumWeights, std::vector<T>& samples)
	{
		unsigned int j = 0;
		for (j = 0; j < weights.size(); j++)
		{
			if (cWeight > weights[j])
			{
				break;
			}
		}
		if (weights.size() < mN)
		{
			sumWeights += cWeight;
			weights.insert(weights.begin() + j, cWeight);
			samples.insert(samples.begin() + j, cSample);
		}
		else
		{
			sumWeights += cWeight;
			weights.insert(weights.begin() + j, cWeight);
			samples.insert(samples.begin() + j, cSample);

			sumWeights -= weights[weights.size() - 1];
			weights.erase(weights.begin() + (weights.size() - 1));
			samples.erase(samples.begin() + (samples.size() - 1));
		}
	}

	bool isItReached(mSampleStructure& pSample)
	{
		if (mNode::isSetAEqualsSetB(pSample.desired_hold_ids, mOptimizer->startState.hold_bodies_ids))
		{
			return true;
		}
		if (pSample.desired_hold_ids[0] == pSample.desired_hold_ids[1] && pSample.desired_hold_ids[1] > -1)
		{
			if (pSample.desired_hold_ids[2] == mOptimizer->startState.hold_bodies_ids[2] && pSample.desired_hold_ids[3] == mOptimizer->startState.hold_bodies_ids[3])
			{
				if (isReachedLegs[0] && isReachedLegs[1])
				{
					return true;
				}
			}
		}
		return false;
	}

	void updateGraphTransitionValues()
	{
		bool flag_path_changed = false;
		for (unsigned int i = 0; i < mTriedPathIndicies.size(); i++)
		{
			int index_on_path = mTriedPathIndicies[i];
			if (!isItReachedFromToOnPath[index_on_path - 1])
			{
				bool is_changed_val = mySamplingGraph.updateGraphNN(mySamplingGraph.getIndexStanceNode(index_on_path-1), mySamplingGraph.getIndexStanceNode(index_on_path));
				if (!flag_path_changed)
				{
					flag_path_changed = is_changed_val;
				}
			}
		}

		if (!flag_path_changed && mySamplingGraph.retPath.size() > 0) // path is reached, A* should return another path
		{
			if (!mSampler::isInSampledStanceSet(mySamplingGraph.retPath, mySamplingGraph.m_found_paths))
			{
				mySamplingGraph.m_found_paths.push_back(mySamplingGraph.retPath);
				mySamplingGraph.setCurPathTriedFather();
			}
		}

		/*for (int i = isItReachedFromToOnPath.size() - 1; i >= 0 && !flag_path_changed; i--)
		{
			int index_on_path = i + 1;
			if (isItReachedFromToOnPath[index_on_path - 1])
			{
				flag_path_changed = mySamplingGraph.updateGraphNN(mySamplingGraph.getIndexStanceNode(index_on_path-1), mySamplingGraph.getIndexStanceNode(index_on_path));
			}
		}*/
		/*for (int i = mNodeIDsForPath.size() - 1; i >= 0 && !flag_path_changed; i--)
		{
			int index_on_path = i;
			if (mNodeIDsForPath[i].size() > 0)
			{
				flag_path_changed = mySamplingGraph.updateGraphNN(mySamplingGraph.getIndexStanceNode(index_on_path-1), mySamplingGraph.getIndexStanceNode(index_on_path));
			}
		}*/

		return;
	}

	bool isTrialStrExistsInQueue(mTrialStructure& nT)
	{
		for (unsigned int i = 0; i < mPriorityQueue.size(); i++)
		{
			mTrialStructure cT = mPriorityQueue[i];
			if (cT._graphNodeId == nT._graphNodeId && cT._graphFatherNodeId == nT._graphFatherNodeId)
			{
				return true;
			}
		}
		return false;
	}

	mSampleStructure retSampleFromGraph(int _fromPathNodeID, int _toPathNodeID)
	{
		std::vector<int> transition_nodes = mNodeIDsForPath[_fromPathNodeID];

		std::vector<mTrialStructure> samples;
		std::vector<float> weights;
		float sumWeights = 0;

		for (unsigned int i = 0; i < transition_nodes.size(); i++)
		{
			mNode* tNodei = &mRRTNodes[transition_nodes[i]];

			std::vector<Vector3> sample_des_points; float mSizeDes = 0;
			Vector3 midDesPoint = mClimberSampler->getHoldStancePosFrom(desiredPath[_toPathNodeID], sample_des_points, mSizeDes);

			float value = sqrt(tNodei->getSumDisEndPosTo(desiredPath[_toPathNodeID], sample_des_points));

			if (tNodei->isInTriedHoldSet(desiredPath[_toPathNodeID]))
			{
				value *= 1000.0f;
			}

			float f = 1 / value;

			createParticleList<mTrialStructure>(10, f, 
				mTrialStructure(tNodei->nodeIndex, mySamplingGraph.getIndexStanceNode(_toPathNodeID), mySamplingGraph.getIndexStanceNode(_toPathNodeID-1), value)
													, weights, sumWeights, samples);
		}

		int rIndex = chooseIndexFromParticles(weights, sumWeights);
		if (rIndex >= 0)
		{
			mSampler::addToSetHoldIDs(_toPathNodeID, mTriedPathIndicies);

			mSampleStructure nSample = mClimberSampler->getSampleFrom(&mRRTNodes[samples[rIndex]._treeNodeId], mySamplingGraph.getStanceGraph(samples[rIndex]._graphNodeId), false);

			isReachedLegs[0] = true;
			isReachedLegs[1] = true;
			if (nSample.initial_hold_ids[0] == -1 && nSample.desired_hold_ids[0] != -1)
			{
				isReachedLegs[0] = false;
			}
			if (nSample.initial_hold_ids[1] == -1 && nSample.desired_hold_ids[1] != -1)
			{
				isReachedLegs[1] = false;
			}

			nSample.toNodeGraph = samples[rIndex]._graphNodeId;
			nSample.fromNodeGraph = samples[rIndex]._graphFatherNodeId;

			printf("\n fNode:%d, tNode:%d \n", nSample.fromNodeGraph, nSample.toNodeGraph);
			return nSample;
		}
		
		return mSampleStructure();
	}

	int chooseIndexFromParticles(std::vector<float>& weights, float sumWeights)
	{
		float preWeight = 0.0f;
		float p_r = mSampler::getRandomBetween_01();
		int cIndex = -1;
		for (unsigned int i = 0; i < weights.size(); i++)
		{
			float cWeight = (weights[i] / sumWeights) + preWeight;
			if (p_r > preWeight && p_r <= cWeight)
			{
				cIndex = i;
				break;
			}
			preWeight = cWeight;
		}
		
		return cIndex;
	}

	bool isStanceExistInTreeNodeChildren(mNode* tNode_n, std::vector<int>& s_i_p1)
	{
		for (unsigned int c = 0; c < tNode_n->mChildrenIndices.size(); c++)
		{
			mNode* cNode = &mRRTNodes[tNode_n->mChildrenIndices[c]];
			if (cNode->isNodeEqualTo(s_i_p1))
			{
				return true;
			}
		}
		return false;
	}

	//////////////////////////////////// utilities for RRT ////////////////////////////////////////////////

	int getBestNodeEnergy(std::vector<int>& tNodes, float& minCost)
	{
		int min_index = -1;
		for (unsigned int i = 0; i < tNodes.size(); i++)
		{
			float cCost = getNodeCost(&mRRTNodes[tNodes[i]], mCaseStudy::Energy);
			if (cCost < minCost)
			{
				minCost = cCost;
				min_index = i;
			}
		}
		return min_index;
	}

	void findBestPathToGoal(mCaseStudy icurrent_case_study)
	{
		Vector3 desiredPos = mContextRRT->getGoalPos();

		int min_index = -1;
		if (goal_nodes.size() == 0)
		{
			float minDis = FLT_MAX;
			for (unsigned int i = 0; i < mRRTNodes.size(); i++)
			{
				mNode nodei = mRRTNodes[i];

				//float cDis = getNodeCost(&nodei);
				float cDis = nodei.getSumDisEndPosTo(desiredPos);
				
				bool isHandConnectedToGoal = isNodeAttachedToGoal(&nodei);

				if (minDis > cDis)
				{
					minDis = cDis;
					min_index = i;
				}
			}

			cCostGoalMoveLimb = getNodeCost(&mRRTNodes[min_index], mCaseStudy::movingLimbs);
		}
		else
		{
			std::vector<int> _sameCostIndices;
			float minCost = FLT_MAX;
			float maxCost = -FLT_MAX;
			switch (icurrent_case_study)
			{
				case mCaseStudy::movingLimbs:
					for (unsigned int i = 0; i < goal_nodes.size(); i++)
					{
						float cCost = getNodeCost(&mRRTNodes[goal_nodes[i]], mCaseStudy::movingLimbs);
						if (cCost < minCost)
						{
							_sameCostIndices.clear();
							_sameCostIndices.push_back(i);
							minCost = cCost;
						}
						else if (cCost == minCost)
						{
							_sameCostIndices.push_back(i);
						}
					}

					cCostGoalMoveLimb = minCost;

					for (unsigned int i = 0; i < _sameCostIndices.size(); i++)
					{
						int index_in_goal_nodes = _sameCostIndices[i];
						float cCost = getNodeCost(&mRRTNodes[goal_nodes[index_in_goal_nodes]], mCaseStudy::Energy);
						if (cCost > maxCost)
						{
							maxCost = cCost;
							cGoalPathIndex = index_in_goal_nodes;
						}
					}

					cCostGoalControlCost = maxCost;

					break;
				case mCaseStudy::Energy:
					cGoalPathIndex = getBestNodeEnergy(goal_nodes, minCost);
					cCostGoalControlCost = minCost;
					cCostGoalMoveLimb = getNodeCost(&mRRTNodes[goal_nodes[cGoalPathIndex]], mCaseStudy::movingLimbs);
					break;
				default:
					break;
			}

			min_index = goal_nodes[cGoalPathIndex];
		}

		std::vector<int> cPath;
		mNode cNode = mRRTNodes[min_index];
		while (cNode.mFatherIndex != -1)
		{
			cPath.insert(cPath.begin(), cNode.nodeIndex);
			if (cNode.mFatherIndex != -1)
				cNode = mRRTNodes[cNode.mFatherIndex];
		}
		cPath.insert(cPath.begin(), cNode.nodeIndex);
		path_nodes_indices = cPath;

		return;
	}

	void playAnimation(mCaseStudy icurrent_case_study)
	{
		clock_t begin = clock();
		if (!mSample.restartSampleStartState)
			mSample.restartSampleStartState = true;

		if (!isPathFound)
		{
			findBestPathToGoal(icurrent_case_study);

			isPathFound = true;

			isAnimationFinished = false;
		}

		if (!isNodeShown)
		{
			if (lastPlayingNode < (int)path_nodes_indices.size() && path_nodes_indices[lastPlayingNode] < (int)mRRTNodes.size())
			{
				mOptimizer->startState = mRRTNodes[path_nodes_indices[lastPlayingNode]].cNodeInfo;
				isNodeShown = true;
				lastPlayingStateInNode = 0;
				lastPlayingNode++;
			}
			else
			{
				// restart showing animation
				lastPlayingNode = 0;
				cTimeElapsed = 0;
				isAnimationFinished = true;
			}
		}
		else
		{
			if (lastPlayingNode < (int)path_nodes_indices.size() && path_nodes_indices[lastPlayingNode] < (int)mRRTNodes.size())
			{
				if (lastPlayingStateInNode < (int)mRRTNodes[path_nodes_indices[lastPlayingNode]].statesFromFatherToThis.size())
				{
					mOptimizer->startState = mRRTNodes[path_nodes_indices[lastPlayingNode]].statesFromFatherToThis[lastPlayingStateInNode];
					lastPlayingStateInNode++;
				}
				else
				{
					isNodeShown = false;
				}
			}
			else
			{
				lastPlayingNode = 0;
				isNodeShown = false;
				cTimeElapsed = 0;
				isAnimationFinished = true;
			}
		}
		Sleep(10); // 1 frame every 30 ms

		mOptimizer->syncMasterContextWithStartState(true);
		clock_t end = clock();
		cTimeElapsed += double(end - begin) / CLOCKS_PER_SEC;
		return;
	}

	void mAcceptOrRejectSample(mSampleStructure& iSample, targetMethod itargetMethd)
	{
		if (rejectCondition(iSample, itargetMethd))
		{
			iSample.isRejected = true;

			if (mOptimizer->startState.hold_bodies_ids[mOptimizer->startState.hold_bodies_ids.size() - 1] != -1 
				|| mOptimizer->startState.hold_bodies_ids[mOptimizer->startState.hold_bodies_ids.size() - 2] != -1)
			{
//				UpdateTree(mController->startState, iSample.closest_node_index, iSample.statesFromTo, iSample);
				UpdateTreeForGraph(mOptimizer->startState, iSample.closest_node_index, iSample.statesFromTo, iSample);
			}

			if (mNode::isSetAEqualsSetB(iSample.desired_hold_ids, mOptimizer->startState.hold_bodies_ids))
			{
				accepted_samples_num++;
			}

			itr_optimization_for_each_sample = 0;
			return;
		}
		if (acceptCondition(iSample))
		{
			iSample.isReached = true;
			accepted_samples_num++;
//			UpdateTree(mController->startState, iSample.closest_node_index, iSample.statesFromTo, iSample);
			UpdateTreeForGraph(mOptimizer->startState, iSample.closest_node_index, iSample.statesFromTo, iSample);
			itr_optimization_for_each_sample = 0;
			return;
		}
	}

	bool rejectCondition(mSampleStructure& iSample, targetMethod itargetMethd)
	{
		if (itargetMethd == targetMethod::Online)
		{
			if (iSample.numItrFixedCost > max_no_improvement_iterations)
				return true;
		}

		if (itr_optimization_for_each_sample > max_itr_optimization_for_each_sample)
			return true;

		if (iSample.isOdeConstraintsViolated)
			return true;

		if (iSample.initial_hold_ids[2] == -1 && iSample.initial_hold_ids[3] == -1)
			return false;

		if (mOptimizer->startState.hold_bodies_ids[2] == mOptimizer->startState.hold_bodies_ids[3] && mOptimizer->startState.hold_bodies_ids[2] == -1)
			return true;

		return false; // the sample is not rejected yet
	}

	bool acceptCondition(mSampleStructure& iSample)
	{
		for (unsigned int i = 0; i < iSample.desired_hold_ids.size(); i++)
		{
			if (mOptimizer->startState.hold_bodies_ids[i] != iSample.desired_hold_ids[i])
				return false; // not reached
		}

		Vector3 mid_point = mOptimizer->startState.getEndPointPosBones(SimulationContext::BodyName::BodyTrunk);
		float cDis = (mSample.destinationP[1] - mid_point).norm();
		if (cDis < 0.01f)
			return true; // is reached and accepted
		return false;
	}

	void m_Connect_Disconnect_ContactPoint(std::vector<int>& desired_holds_ids)
	{
		float min_reject_angle = (PI / 2) - (0.3f * PI);
		//float max_acceptable_angle = 1.3f * PI;

		for (unsigned int i = 0; i < desired_holds_ids.size(); i++)
		{
			if (desired_holds_ids[i] != -1)
			{
				Vector3 hold_pos_i = mClimberSampler->getHoldPos(desired_holds_ids[i]);
				Vector3 contact_pos_i = mOptimizer->startState.getEndPointPosBones(SimulationContext::ContactPoints::LeftLeg + i);

				float dis_i = (hold_pos_i - contact_pos_i).norm();

				if (dis_i < 0.25f * holdSize)
				{
					if (i <= 1) // left leg and right leg
					{
						Vector3 dir_contact_pos = -mOptimizer->startState.getBodyDirectionZ(SimulationContext::ContactPoints::LeftLeg + i);
						//mController->startState.bodyStates[SimulationContext::ContactPoints::LeftLeg + i].pos;
						float m_angle_btw = SimulationContext::getAbsAngleBtwVectors(-Vector3(0.0f, 0.0f, 1.0f), dir_contact_pos);

						if (m_angle_btw > min_reject_angle)
						{
							isReachedLegs[i] = true;
							if (i == 0 && mOptimizer->startState.hold_bodies_ids[1] > -1 && mOptimizer->startState.hold_bodies_ids[1] == desired_holds_ids[0])
							{
								continue;
							}
							if (i == 1 && mOptimizer->startState.hold_bodies_ids[0] > -1 && mOptimizer->startState.hold_bodies_ids[0] == desired_holds_ids[1])
							{
								continue;
							}
							mOptimizer->startState.hold_bodies_ids[i] = desired_holds_ids[i];
						}
						else
						{
							isReachedLegs[i] = false;
							mOptimizer->startState.hold_bodies_ids[i] = -1;
						}
					}
					else
					{
						mOptimizer->startState.hold_bodies_ids[i] = desired_holds_ids[i];
					}
				}
				else if (dis_i > 0.5f * holdSize)
				{
					if (i <= 1)
					{
						isReachedLegs[i] = false;
					}
					mOptimizer->startState.hold_bodies_ids[i] = -1;
				}
			}
			else
			{
				if (i <= 1)
				{
					isReachedLegs[i] = false;
				}
				mOptimizer->startState.hold_bodies_ids[i] = -1;
			}
		}
	}

	void mSteerFunc(bool isOnlineOptimization)
	{
		mOptimizer->syncMasterContextWithStartState(itr_optimization_for_each_sample < max_waste_interations || !isOnlineOptimization);
		if (optimizerType==otCMAES && !useOfflinePlanning)
			Debug::throwError("CMAES cannot be used in online mode!");
		if (optimizerType==otCPBP)	
		{
			if (itr_optimization_for_each_sample==0 && useOfflinePlanning)
				mOptimizer->reset();
			mOptimizer->optimize_the_cost(isOnlineOptimization, mSample.sourceP, mSample.destinationP, mSample.desired_hold_ids, true);
		}
		/*else 
			mController->optimize_the_cost_cmaes(itr_optimization_for_each_sample==0,mSample.sourceP, mSample.destinationP, mSample.destinationA);*/

		// check changing of the cost
		float costImprovement = std::max(0.0f,mSample.cOptimizationCost-mOptimizer->current_cost_state);
		rcPrintString("Traj. cost improvement: %f",costImprovement);
		if (costImprovement < noCostImprovementThreshold)
		{
			mSample.numItrFixedCost++;
		}
		else
		{
			mSample.numItrFixedCost = 0;
			
		}
		mSample.cOptimizationCost = mOptimizer->current_cost_state;

		//apply the best control to get the start state of next frame
		//advance_time is ture for the online optimization
		if (isOnlineOptimization) 
		{
			mStepOptimization(0, isOnlineOptimization);
		}
		return;
	}

	void mStepOptimization(int cTimeStep, bool debugPrint = false)
	{
		std::vector<BipedState> nStates;
		
		bool flagAddSimulation = mOptimizer->simulateBestTrajectory(itr_optimization_for_each_sample >= max_waste_interations, mSample.desired_hold_ids, nStates);
			
			//advance_simulation(cTimeStep, debugPrint,itr_optimization_for_each_sample >= max_waste_interations, nStates);

		for (unsigned int ns = 0; ns < nStates.size(); ns++)
		{
			BipedState nState = nStates[ns];
			mSample.statesFromTo.push_back(nState);
		}
		mSample.control_cost += mOptimizer->current_cost_control;

		if (!flagAddSimulation)
		{
			mSample.isOdeConstraintsViolated = true;
		}
	}

	void setArmsLegsHoldPoses()
	{
		mOptimizer->startState.hold_bodies_ids = mSample.initial_hold_ids;
		for (unsigned int i = 0; i < mOptimizer->startState.hold_bodies_ids.size(); i++)
		{
			if (mOptimizer->startState.hold_bodies_ids[i] != -1)
			{
				Vector3 hPos_i = mClimberSampler->getHoldPos(mOptimizer->startState.hold_bodies_ids[i]);
				Vector3 endPos_i = mOptimizer->startState.getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg + i);
				float cDis = (hPos_i - endPos_i).norm();
				if (cDis > 0.5f * holdSize)
				{
					mOptimizer->startState.hold_bodies_ids[i] = -1;
				}
			}
		}
		return;
	}

	////////////////////////////////// utilities for RRT nodes////////////////////////////////////////////

	std::vector<double> getNodeKeyFrom(BipedState& c)
	{
		std::vector<double> rKey;

		for (unsigned int i = 0; i < c.bodyStates.size(); i++)
		{
			BodyState body_i = c.bodyStates[i];
			rKey.push_back(body_i.getPos().x());
			rKey.push_back(body_i.getPos().y());
			rKey.push_back(body_i.getPos().z());
			//rKey.push_back(body_i.angle);

			//rKey.push_back(body_i.vel.x);
			//rKey.push_back(body_i.vel.y);
			//rKey.push_back(body_i.aVel);
		}

		return rKey;
	}

	int getNearestNode(BipedState& c)
	{
		int index_node = mKDTreeNodeIndices.nearest(getNodeKeyFrom(c));

		return index_node;
	}

	float getNodeCost(mNode* iNode, mCaseStudy icurrent_case_study)
	{
		int cIndex = iNode->nodeIndex;
		float cCost = 0.0f;

		int mCounter = 0;

		while (cIndex >= 0)
		{
			switch (icurrent_case_study)
			{
			case mCaseStudy::movingLimbs:
				cCost += mRRTNodes[cIndex].cCost;
				break;
			case mCaseStudy::Energy:
				cCost += mRRTNodes[cIndex].control_cost;
				break;
			}

			cIndex = mRRTNodes[cIndex].mFatherIndex;

			mCounter++;
			if (mCounter > (int)mRRTNodes.size())
			{
				break;
			}
		}

		return cCost;
	}

	bool isBodyStateEqual(mNode* nearest_node, BipedState& c)
	{
		float pos_err = 0.01f;
		float angle_err = 0.01f;
		float ang_vel_err = 0.01f;
		float vel_err = 0.01f;
		for (unsigned int i = 0; i < c.bodyStates.size(); i++)
		{
			if ((nearest_node->cNodeInfo.bodyStates[i].getPos() - c.bodyStates[i].getPos()).norm() > pos_err)
			{
	//			printf("position: %f", (nearest_node->cNodeInfo.bodyStates[i].getPos() - c.bodyStates[i].getPos()).norm());
				return false;
			}
			if ((nearest_node->cNodeInfo.bodyStates[i].getAngle() - c.bodyStates[i].getAngle()).norm() > angle_err)
			{
	//			printf("angle: %f", (nearest_node->cNodeInfo.bodyStates[i].getAngle() - c.bodyStates[i].getAngle()).norm());
				return false;
			}
			if ((nearest_node->cNodeInfo.bodyStates[i].getAVel() - c.bodyStates[i].getAVel()).norm() > ang_vel_err)
			{
	//			printf("%f", (nearest_node->cNodeInfo.bodyStates[i].getAngle() - c.bodyStates[i].getAngle()).norm());
				return false;
			}
			if ((nearest_node->cNodeInfo.bodyStates[i].getVel() - c.bodyStates[i].getVel()).norm() > vel_err)
			{
		//		printf("%f", (nearest_node->cNodeInfo.bodyStates[i].getAngle() - c.bodyStates[i].getAngle()).norm());
				return false;
			}
		}
		return true;
	}

	void AddToTransitions(Vector2 nT)
	{
		bool flag_exists = false;
		for (unsigned int j = 0; j < mTransitionFromTo.size() && !flag_exists; j++)
		{
			Vector2 oT = mTransitionFromTo[j];
			if ((oT - nT).norm() < 0.1f)
			{
				flag_exists = true;
			}
		}
		if (!flag_exists)
		{
			mTransitionFromTo.push_back(nT);
			numPossRefine++;
		}
	}

	void UpdateTreeForGraph(BipedState& c, int iFatherIndex, std::vector<BipedState>& _fromFatherToThis, mSampleStructure& iSample)
	{
		int indexNearestNode = getNearestNode(c);
		if (indexNearestNode == -1)
		{
			addNode(c, iFatherIndex, _fromFatherToThis, iSample);
			return;
		}
		mNode* closest_node = &mRRTNodes[indexNearestNode];
		if (isBodyStateEqual(closest_node, c)) //then if they are actually equal, rewire
		{
			iSample.toNode = closest_node->nodeIndex;
			return;
		}

		if (!initialstance_exists_in_tree)
		{
			if (mNode::isSetAEqualsSetB(initial_stance, c.hold_bodies_ids))
			{
				initialstance_exists_in_tree = true;
			}
		}

		int nNodeIndex = addNode(c, iFatherIndex, _fromFatherToThis, iSample);

		mNode nNode = mRRTNodes[nNodeIndex];
		
		if (initialstance_exists_in_tree && index_node_initial_stance < 0)
		{
			index_node_initial_stance = mRRTNodes.size() - 1;
		}

		// check goal is reached or not
		if (!isGoalReached)
		{
			isGoalReached = isNodeAttachedToGoal(&nNode);
		}
		return;
	}

	int addNode(BipedState& c, int iFatherIndex, std::vector<BipedState>& _fromFatherToThis, mSampleStructure& iSample)
	{
		mNode nNode = mNode(c.getNewCopy(mContextRRT->getNextFreeSavingSlot(), mContextRRT->getMasterContextID()), iFatherIndex, mRRTNodes.size(), _fromFatherToThis);

		nNode.control_cost = iSample.control_cost;

		if (iFatherIndex != -1)
		{
			nNode.cCost = nNode.getCostNumMovedLimbs(mRRTNodes[iFatherIndex].cNodeInfo.hold_bodies_ids);
		}

		nNode.poss_hold_ids = c.hold_bodies_ids;
		if (iSample.desired_hold_ids.size() > 0)
		{
			if (isReachedLegs[0])
			{
				nNode.poss_hold_ids[0] = iSample.desired_hold_ids[0];
			}
			if (isReachedLegs[1])
			{
				nNode.poss_hold_ids[1] = iSample.desired_hold_ids[1];
			}
		}

		mRRTNodes.push_back(nNode);

		if (iFatherIndex != -1)
		{
			mRRTNodes[iFatherIndex].mChildrenIndices.push_back(mRRTNodes.size() - 1);
		}

		mKDTreeNodeIndices.insert(getNodeKeyFrom(c), mRRTNodes.size() - 1);

		if (isNodeAttachedToGoal(&nNode))
		{
			int cNodeIndex = nNode.nodeIndex;
			while (cNodeIndex != -1)
			{
				mNode& cNode = mRRTNodes[cNodeIndex];
				if (mNode::isSetAEqualsSetB(cNode.cNodeInfo.hold_bodies_ids, initial_stance))
				{
					mSampler::addToSetHoldIDs(nNode.nodeIndex, goal_nodes);
					break;
				}
				cNodeIndex = cNode.mFatherIndex;
			}
		}

		iSample.toNode = nNode.nodeIndex;

		return nNode.nodeIndex;
	}

	bool isNodeAttachedToGoal(mNode* nodei)
	{
		if (nodei->cNodeInfo.hold_bodies_ids[3] != -1)
		{
			Vector3 rightHandPos = mClimberSampler->getHoldPos(nodei->cNodeInfo.hold_bodies_ids[3]);
			if ((mContextRRT->getGoalPos() - rightHandPos).norm() < 0.3f * holdSize)
			{
				return true;
			}
		}
		if (nodei->cNodeInfo.hold_bodies_ids[2] != -1)
		{
			Vector3 leftHandPos = mClimberSampler->getHoldPos(nodei->cNodeInfo.hold_bodies_ids[2]);
			if ((mContextRRT->getGoalPos() - leftHandPos).norm() < 0.3f * holdSize)
			{
				return true;
			}
		}
		return false;
	}

	/////////////////////////////////////////// common utilities //////////////////////////////////////////

	Vector3 uniformRandomPointBetween(Vector3 iMin, Vector3 iMax)
	{
		//Random between Min, Max
		float r1 = mSampler::getRandomBetween_01();
		float r2 = mSampler::getRandomBetween_01();
		float r3 = mSampler::getRandomBetween_01();
		Vector3 dis = iMax - iMin;

		return iMin + Vector3(dis.x() * r1, dis.y() * r2, dis.z() * r3); 
	}

}* myRRTPlanner;

class mTestControllerClass
{
public:

	class mSavedDataState
	{
	public:
		BipedState _bodyState;
		std::vector<int> _desired_holds_ids;
		std::vector<BipedState> _fromToMotions;
	};

	class mSavedUserStudyData
	{
	public:
		std::vector<int> _from_holds_ids;
		std::vector<int> _to_holds_ids;
		float _starting_time;
		float _end_time;
		bool _succedd;
		bool _isBackSpaceHit;

		std::string toString()
		{
			//#x,y,z,f,dx,dy,dz,k,s
			std::string write_buff;
			char _buff[200];
			sprintf_s(_buff, "%d,", _from_holds_ids[0]); write_buff += _buff;
			sprintf_s(_buff, "%d,", _from_holds_ids[1]); write_buff += _buff;
			sprintf_s(_buff, "%d,", _from_holds_ids[2]); write_buff += _buff;
			sprintf_s(_buff, "%d,", _from_holds_ids[3]); write_buff += _buff;

			sprintf_s(_buff, "%d,", _to_holds_ids[0]); write_buff += _buff;
			sprintf_s(_buff, "%d,", _to_holds_ids[1]); write_buff += _buff;
			sprintf_s(_buff, "%d,", _to_holds_ids[2]); write_buff += _buff;
			sprintf_s(_buff, "%d,", _to_holds_ids[3]); write_buff += _buff;

			sprintf_s(_buff, "%f,", _starting_time); write_buff += _buff;
			sprintf_s(_buff, "%f,", _end_time); write_buff += _buff;

			sprintf_s(_buff, "%d,", _succedd ? 1 : 0); write_buff += _buff;

			sprintf_s(_buff, "%d\n", _isBackSpaceHit ? 1 : 0); write_buff += _buff;

			return write_buff;
		}
	};
	float timeSpendOnAnimation;
	float timeSpendInPlanning;
	float preVisitingTime;
	bool isBackSpaceHit;

	std::vector<mSavedUserStudyData> _savedUserData;

	void mSaveUserDataFunc()
	{
		mSavedUserStudyData cData;
		cData._from_holds_ids = mSample.initial_hold_ids;
		cData._to_holds_ids = mSample.desired_hold_ids;

		cData._starting_time = mSample.starting_time;
		cData._end_time = timeSpendInPlanning; // one event happened (backspace, reached, failed, change)

		cData._succedd = mNode::isSetAEqualsSetB(mOptimizer->startState.hold_bodies_ids, mSample.desired_hold_ids);

		cData._isBackSpaceHit = isBackSpaceHit;

		_savedUserData.push_back(cData);
	}

	mSampler* mClimberSampler;
	mController* mOptimizerOnline;
	mController* mOptimizerOffline;
	SimulationContext* mContextRRT;
	mSampleStructure mSample;

	//handling offline - online method
	enum targetMethod{Offiline = 0, Online = 1};

	std::vector<int> desired_holds_ids; // for hold ids for the hands and feet of the agent {0: ll, 1: rl, 2: lh, 3: rh}

	// handling falling problem of the climber
	std::list<mSavedDataState> lOptimizationBipedStates;
	std::list<int> deletedSavingSlots;
	std::vector<BipedState> _fromToMotions;

	int itr_optimization_for_each_sample;
	int max_itr_optimization_for_each_sample;
	int max_no_improvement_iterations;

	bool pauseOptimization;

	int currentIndexOfflineState;

	////////////////////////////////////////////////////// mouse info /////////////////////////////////////////////////

	bool update_hold;
	float disFromCOM;

	// comes from mouse function
	Vector3 rayBegin, rayEnd;
	float rayDis;
	int bMouse;
	int cX, cY;//current x, y
	int lX, lY;//last x, y

	float theta_torso;
	float phi_torso; // we do not need this value (phi_torso = 0)

	// we are looking at climber's center of mass 
	Vector3 lookDir;

	int selected_body; // contains hit geom of humanoid climber
	int selected_hold; // contains hit geom of hold on the wall

	int state_selection; // {0: for selecting humanoid body,1: for selecting hold}
	int last_hit_geom_id;

	int currentIndexBody;

	bool isOptDone;
	bool isRestartedForOfflineOpt;

	bool debugShow;

	// for playing animation
	
	unsigned int indexOfSavedStates;
	unsigned int index_in_currentState;
	int showing_state;
	BipedState lastStateBeforeAnimation;

	void play_animation()
	{
		if (showing_state == 1)
		{
			if (index_in_currentState < _fromToMotions.size())
			{
				mOptimizerOnline->startState = _fromToMotions[index_in_currentState];
				index_in_currentState++;
			}
			else
			{
				showing_state = 2;
			}
		}

		if (showing_state == 0 || showing_state == 2)
		{
			std::list<mSavedDataState>::iterator pointerToSavedStates = lOptimizationBipedStates.begin();
			for (unsigned int counter = 0; counter < indexOfSavedStates && showing_state == 0; counter++)
			{
				if (pointerToSavedStates != lOptimizationBipedStates.end())
					pointerToSavedStates++;
				else
				{
					indexOfSavedStates = lOptimizationBipedStates.size() - 1;
					break;
				}
			}

			if (index_in_currentState < pointerToSavedStates->_fromToMotions.size() && showing_state == 0)
			{
				mOptimizerOnline->startState = pointerToSavedStates->_fromToMotions[index_in_currentState];
				index_in_currentState++;
			}
			else
			{
				index_in_currentState = 0;
				if (showing_state == 0)
				{
					if (indexOfSavedStates < lOptimizationBipedStates.size() - 1)
					{
						indexOfSavedStates++;
					}
					else if (_fromToMotions.size() > 0)
					{
						showing_state = 1;
					}
					else
					{
						indexOfSavedStates = 0;
						showing_state = 0;
					}
				}
				else
				{
					indexOfSavedStates = 0;
					showing_state = 0;
				}
			}
		}
		//Sleep(60);
		mOptimizerOnline->syncMasterContextWithStartState(true);
		return;
	}

	mTestControllerClass(SimulationContext* iContextRRT, mController* iOnlineController, mController* iOfflineController, mSampler* iHoldSampler)
	{
		timeSpendInPlanning = 0;
		timeSpendOnAnimation = 0;
		preVisitingTime = 0;
		isBackSpaceHit = false;

		mOptimizerOnline = iOnlineController;
		mOptimizerOffline = iOfflineController;
		mContextRRT = iContextRRT;
		mClimberSampler = iHoldSampler;

		for (unsigned int i = 0; i < mOptimizerOnline->startState.hold_bodies_ids.size(); i++)
		{
			mSample.initial_hold_ids.push_back(mOptimizerOnline->startState.hold_bodies_ids[i]);
			mSample.desired_hold_ids.push_back(mOptimizerOnline->startState.hold_bodies_ids[i]);
			desired_holds_ids.push_back(-1);
		}

		mSavedDataState ndata;
		ndata._bodyState = mOptimizerOnline->startState.getNewCopy(mContextRRT->getNextFreeSavingSlot(), mContextRRT->getMasterContextID());
		ndata._desired_holds_ids = desired_holds_ids;

		lOptimizationBipedStates.push_back(ndata);

		//the online method is not working good with 3*(int)(cTime/nPhysicsPerStep), its value should be around 50
		max_itr_optimization_for_each_sample = useOfflinePlanning ? 10*(int)(cTime/nPhysicsPerStep) : (int)(1.5f * cTime);
		itr_optimization_for_each_sample = 0;
		//the online method is not working good with (int)(cTime/nPhysicsPerStep)/4, its value should be around 8
		max_no_improvement_iterations = useOfflinePlanning ? 20 : (int)(cTime/4); //max_itr_optimization_for_each_sample

		update_hold = false;
		disFromCOM = 5.0f;

		rayDis = 0.0f;
		bMouse = -1;
		selected_body = -1;
		selected_hold = -1;

		state_selection = 0;
		last_hit_geom_id = -1;

		currentIndexBody = 0;

		cX = 0; cY = 0; lX = 0; lY = 0;
		resetTorsoDir();

		lookDir = Vector3(0.0f,0.0f,0.0f);

		pauseOptimization = true;
		currentIndexOfflineState = 0;
		isOptDone = false;
		isRestartedForOfflineOpt = false;

		mOptimizerOnline->syncMasterContextWithStartState(true);

		debugShow = false;

		// for playing animation
		index_in_currentState = 0;
		indexOfSavedStates = 0;
		lastStateBeforeAnimation = mOptimizerOnline->startState.getNewCopy(mContextRRT->getNextFreeSavingSlot(), mContextRRT->getMasterContextID());
		showing_state = 0;
	}

	~mTestControllerClass()
	{
		if (CreateDirectory("ClimberResults", NULL) || ERROR_ALREADY_EXISTS == GetLastError())
		{
			mFileHandler writeFile(mContextRRT->getAppendAddress("ClimberResults\\OutRoute", mContextRRT->_RouteNum + 1, ".txt"));

			writeFile.openFileForWritingOn();

			std::string write_buff;
			char _buff[100];
			sprintf_s(_buff, "%f,%f\n", timeSpendInPlanning, timeSpendOnAnimation); write_buff += _buff;
			writeFile.writeLine(write_buff);
			for (unsigned int i = 0; i < _savedUserData.size(); i++)
			{
				std::string cData = _savedUserData[i].toString();
				writeFile.writeLine(cData);
			}

			writeFile.mCloseFile();
		}
	}

	void resetTorsoDir()
	{
		theta_torso = 90;
		phi_torso = 0;
	}

	void runLoopTest(bool advance_time, bool playAnimation, float current_time)
	{
		targetMethod itargetMethd = useOfflinePlanning ? targetMethod::Offiline : targetMethod::Online; 

		float delta_time = current_time - preVisitingTime;
		preVisitingTime = current_time;

		if (!playAnimation)
		{
			timeSpendInPlanning += delta_time;
			
			mOptimizerOnline->startState = lastStateBeforeAnimation;
			mOptimizerOffline->startState = lastStateBeforeAnimation;

			trackMouse();
			updateSampleInfo();
			if (!pauseOptimization)
			{
				if (itargetMethd == targetMethod::Online)
				{
					runTestOnlineOpt(advance_time);
				}
				else
				{
					runTestOfflineOpt(advance_time);
				}
			}
			else
			{
				if (!useOfflinePlanning)
					mOptimizerOnline->syncMasterContextWithStartState(true);
				else
					mOptimizerOffline->syncMasterContextWithStartState(true);
			}

			// play best simulated trajectory after the optimization is done
			if (itargetMethd == targetMethod::Offiline)
			{		
				if (isOptDone)
				{
					std::vector<BipedState> nStates;
					isOptDone = !mOptimizerOffline->simulateBestTrajectory(true, mSample.desired_hold_ids, nStates);
					
					for (unsigned int i = 0; i < nStates.size(); i++)
					{
						_fromToMotions.push_back(nStates[i]);
					}

					if (!isOptDone)
					{
						mSample.numItrFixedCost = 0;
						itr_optimization_for_each_sample = 0;
					}
				}
			}
			if (!useOfflinePlanning)
				lastStateBeforeAnimation = mOptimizerOnline->startState.getNewCopy(lastStateBeforeAnimation.saving_slot_state, mContextRRT->getMasterContextID());
			else
				lastStateBeforeAnimation = mOptimizerOffline->startState.getNewCopy(lastStateBeforeAnimation.saving_slot_state, mContextRRT->getMasterContextID());
		}
		else
		{
			play_animation();
			timeSpendOnAnimation += delta_time;
		}

		if (!playAnimation)
		{
			if (mOptimizerOnline->startState.hold_bodies_ids[2] == -1 && mOptimizerOnline->startState.hold_bodies_ids[3] == -1)
			{
				useOfflinePlanning = true;
				if (!isRestartedForOfflineOpt)
				{
					isRestartedForOfflineOpt = true;

					mSample.numItrFixedCost = 0;
					mSample.cOptimizationCost = FLT_MAX;
					itr_optimization_for_each_sample = 0;
					mSample.isReached = false;
					mSample.isRejected = false;
				}
			}
			else
			{
				if (useOfflinePlanning && !isOptDone)
				{
					useOfflinePlanning = false;
					isRestartedForOfflineOpt = false;
				}
			}
			max_itr_optimization_for_each_sample = useOfflinePlanning ? 10*(int)(cTime/nPhysicsPerStep) : (int)(1.5f * cTime);
			max_no_improvement_iterations = useOfflinePlanning ? 20 : (int)(cTime/4);
		}

		drawDesiredLines(playAnimation);
		printClimberInfo();
	}

	void removeLastSavingState(bool playAnimation)
	{
		if (isOptDone || playAnimation)
		{
			// we are playing the animation
			return;
		}

		isBackSpaceHit = true;
		//mSample.starting_time = timeSpendInPlanning;
		if (mSample.isReached || mSample.isRejected || !pauseOptimization) // if new state is about to get added
		{
			// get back to the last saved state
			mOptimizerOnline->startState = lOptimizationBipedStates.back()._bodyState;
			mOptimizerOffline->startState = lOptimizationBipedStates.back()._bodyState;
			desired_holds_ids = lOptimizationBipedStates.back()._desired_holds_ids;

			//bool saved_new_slot = saveLastReachedState();
			//if (!saved_new_slot)
			_fromToMotions.clear();
			/*if (lOptimizationBipedStates.size() > 1 && pauseOptimization)
			{
				int deletedSlot = lOptimizationBipedStates.back()._bodyState.saving_slot_state;

				deletedSavingSlots.push_back(deletedSlot);
				lOptimizationBipedStates.pop_back();
			}*/
		}
		else
		{
			if (lOptimizationBipedStates.size() > 1)
			{
				int deletedSlot = lOptimizationBipedStates.back()._bodyState.saving_slot_state;

				deletedSavingSlots.push_back(deletedSlot);
				lOptimizationBipedStates.pop_back();
			}

			mOptimizerOnline->startState = lOptimizationBipedStates.back()._bodyState;
			mOptimizerOffline->startState = lOptimizationBipedStates.back()._bodyState;
			desired_holds_ids = lOptimizationBipedStates.back()._desired_holds_ids;
		}

		if (indexOfSavedStates > lOptimizationBipedStates.size() - 1)
		{
			indexOfSavedStates = lOptimizationBipedStates.size() - 1;
		}

		pauseOptimization = true;
		mSample.isReached = false;
		mSample.isRejected = false;
		itr_optimization_for_each_sample = 0;
		mSample.numItrFixedCost = 0;
		mSample.isOdeConstraintsViolated = false;
		mSample.statesFromTo.clear();
		mSample.cOptimizationCost = FLT_MAX;

		if (!useOfflinePlanning)
			mOptimizerOnline->syncMasterContextWithStartState(true);
		else
			mOptimizerOffline->syncMasterContextWithStartState(true);

		if (!useOfflinePlanning)
			lastStateBeforeAnimation = mOptimizerOnline->startState.getNewCopy(lastStateBeforeAnimation.saving_slot_state, mContextRRT->getMasterContextID());
		else
			lastStateBeforeAnimation = mOptimizerOffline->startState.getNewCopy(lastStateBeforeAnimation.saving_slot_state, mContextRRT->getMasterContextID());

		return;
	}

	void runOptimization()
	{
		if (!pauseOptimization)
		{
			return;
		}

		saveLastReachedState();

		if (pauseOptimization)
		{
			mSample.starting_time = timeSpendInPlanning;
		}

		pauseOptimization = false;
		mSample.isReached = false;
		mSample.isRejected = false;
	}

	void updateCameraPositionByMouse(float& theta, float& phi, bool showDebug = false)
	{
		int deltax = cX - lX;
		int deltay = cY - lY;
		if (showDebug)
			rcPrintString("button mouse: %d, deltax: %d, deltay: %d", bMouse, deltax, deltay);
		if (bMouse == 4) //right click
		{
			theta += float (deltax) * 0.5f;
			phi -= float (deltay) * 0.5f;

			deltax = 0;
			deltay = 0;

			lX = cX;
			lY = cY;
		}
		return;
	}

	static Vector3 getDirectionFromAngles(float theta, float phi)
	{
		Vector3 dir;
		dir[0] = (float) (cosf (theta*DEG_TO_RAD) * cosf (phi*DEG_TO_RAD));
		dir[1] = (float) (sinf (theta*DEG_TO_RAD) * cosf (phi*DEG_TO_RAD));
		dir[2] = (float) (sinf (phi*DEG_TO_RAD));
		dir.normalize();
		return dir;
	}

private:
	bool saveLastReachedState()
	{
		//save last state if we moved to any desired position
		if (mSample.isReached || mSample.isRejected)
		{
			// save last optimization state
			int nSavingSlot = -1;
			if (deletedSavingSlots.size() > 0)
			{
				nSavingSlot = deletedSavingSlots.front();
				deletedSavingSlots.pop_front();
			}
			else
			{
				nSavingSlot = mContextRRT->getNextFreeSavingSlot();
			}

			mSavedDataState ndata;
			if (!useOfflinePlanning)
				ndata._bodyState = mOptimizerOnline->startState.getNewCopy(nSavingSlot, mContextRRT->getMasterContextID());
			else
				ndata._bodyState = mOptimizerOffline->startState.getNewCopy(nSavingSlot, mContextRRT->getMasterContextID());

			ndata._desired_holds_ids = desired_holds_ids;
			ndata._fromToMotions = _fromToMotions;

			lOptimizationBipedStates.push_back(ndata);

			if (showing_state == 1)
			{
				showing_state = 0;
				indexOfSavedStates = lOptimizationBipedStates.size() - 1;
			}

			_fromToMotions.clear();

			return true;
		}

		//_fromToMotions.clear();
		lOptimizationBipedStates.back()._desired_holds_ids = desired_holds_ids;
		return false;
	}

	/////////////////////////////////////////////// choose offline or online mode
	void runTestOfflineOpt(bool advance_time)
	{
		// sync all contexts with stating state and current stance
		mOptimizerOffline->syncMasterContextWithStartState(true);

		if (itr_optimization_for_each_sample == 0)
			mOptimizerOffline->reset();

		if (optimizerType == otCMAES)
			mOptimizerOffline->optimize_the_cost(itr_optimization_for_each_sample == 0, mSample.sourceP, mSample.destinationP, mSample.desired_hold_ids, debugShow);
		else
			mOptimizerOffline->optimize_the_cost(false, mSample.sourceP, mSample.destinationP, mSample.desired_hold_ids, debugShow);

		if (advance_time)
		{
			float costImprovement = std::max(0.0f,mSample.cOptimizationCost-mOptimizerOffline->current_cost_state);
			rcPrintString("Traj. cost improvement: %f",costImprovement);
			if (costImprovement < noCostImprovementThreshold)
			{
				mSample.numItrFixedCost++;
			}
			else
			{
				mSample.numItrFixedCost = 0;
			
			}
			if (mOptimizerOffline->current_cost_state < mSample.cOptimizationCost)
				mSample.cOptimizationCost = mOptimizerOffline->current_cost_state;

			itr_optimization_for_each_sample++;
		}
		// offline optimization is done, simulate forward on the best trajectory
		if ((advance_time && mSample.numItrFixedCost > max_no_improvement_iterations)
			|| itr_optimization_for_each_sample > max_itr_optimization_for_each_sample)
		{
			pauseOptimization = true;
			isOptDone = true;
			mSample.isReached = true;
			mSaveUserDataFunc();
		}
		else
		{
			isOptDone = false;
			mSample.isReached = false;
		}
		return;
	}

	void runTestOnlineOpt(bool advance_time)
	{
		mSteerFunc(advance_time);

		if (advance_time)
		{
			itr_optimization_for_each_sample++;
		}

		if (itr_optimization_for_each_sample > max_itr_optimization_for_each_sample ||
			mSample.numItrFixedCost > max_no_improvement_iterations)
		{
			itr_optimization_for_each_sample = 0;
			mSample.numItrFixedCost = 0;
			mSample.isOdeConstraintsViolated = false;
			mSample.isReached = true;
		}

		if (mSample.isReached || mSample.isRejected)
		{
			pauseOptimization = true;
			mSaveUserDataFunc();
		}

		return;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	void printClimberInfo()
	{
		rcPrintString("climber's height is: %f \n", mContextRRT->climberHeight);
		return;
	}

	void drawDesiredLines(bool playAnimation)
	{
		mSample.drawDesiredTorsoDir(mContextRRT->getBonePosition(SimulationContext::BodyName::BodyTrunk));

		if (!playAnimation)
		{
			rcSetColor(0,1,0,1);
			for (unsigned int i = 0; i < 4; i++)
			{
				if (mSample.desired_hold_ids[i] != -1)
				{
					Vector3 toPoint = mClimberSampler->getHoldPos(mSample.desired_hold_ids[i]);
					float _s = mContextRRT->getHoldSize(mSample.desired_hold_ids[i]);
					toPoint[1] -= (_s / 2);
					SimulationContext::drawLine(mContextRRT->getEndPointPosBones((int)(SimulationContext::BodyName::BodyLeftLeg + i)), toPoint);
				}
				else
				{
					SimulationContext::drawCross(mContextRRT->getEndPointPosBones((int)(SimulationContext::BodyName::BodyLeftLeg + i)));
				}
			}
		}
		if (pauseOptimization && debugShow)
			mOptimizer->visualizeForceDirections();
		return;
	}

	void trackMouse(bool showDebugMouseInfo = false)
	{
		if (showDebugMouseInfo)
		{
			// mouse info
			rcPrintString("bX:%f, bY:%f, bZ:%f \n",rayBegin.x(), rayBegin.y(), rayBegin.z());
			rcPrintString("eX:%f, eY:%f, eZ:%f, B:%d \n",rayEnd.x(), rayEnd.y(), rayEnd.z(), bMouse);

			rcSetColor(1,0,0);
			Vector3 rayStart = rayBegin + lookDir;
			SimulationContext::drawLine(rayStart, rayEnd);
		}

		Vector3 rayDir = (rayEnd - rayBegin).normalized();
		rayDis = (rayEnd - rayBegin).norm();

		dVector3 out_pos;
		float out_depth;
		int hitGeomID = -1;
		int cIndexBody = -1;

		if (!this->update_hold)
		{
			if (bMouse != 1) // any click other than left click
			{
				hitGeomID = odeRaycastGeom(rayBegin.x(), rayBegin.y(), rayBegin.z(),
									   rayDir.x(), rayDir.y(), rayDir.z(), 
									   rayDis, out_pos, out_depth,
									   unsigned long(0x7FFF), unsigned long(0x7FFF));

				cIndexBody = (int)mContextRRT->getIndexHandsAndLegsFromGeom(selected_body);
				if (state_selection == 1 && selected_body >= 0)
				{
					selected_hold = last_hit_geom_id;
					state_selection = 0;

					if (cIndexBody >= (int)SimulationContext::MouseTrackingBody::MouseLeftLeg && cIndexBody <= (int)SimulationContext::MouseTrackingBody::MouseRightHand)
					{
						currentIndexBody = cIndexBody;
						desired_holds_ids[currentIndexBody] = mContextRRT->getIndexHoldFromGeom(selected_hold);
						if (desired_holds_ids[currentIndexBody] < 0)
							desired_holds_ids[currentIndexBody] = -1;
					}
				}
				else
				{
					selected_hold = -1;
					state_selection = 0;
				}
				if (mContextRRT->getIndexHandsAndLegsFromGeom(hitGeomID)>= 0)
					selected_body = hitGeomID;
				else
					selected_body = -1;
			}
			else if (bMouse == 1) // left click
			{
				hitGeomID = odeRaycastGeom(rayBegin.x(), rayBegin.y(), rayBegin.z(),
									   rayDir.x(), rayDir.y(), rayDir.z(), 
									   rayDis, out_pos, out_depth,
									   unsigned long(0x8000), unsigned long(0x8000));

				if (state_selection == 0)
				{
					state_selection = 1;
				}
				cIndexBody = (int)mContextRRT->getIndexHandsAndLegsFromGeom(selected_body);
				if (cIndexBody >= (int)SimulationContext::MouseTrackingBody::MouseLeftLeg && cIndexBody <= (int)SimulationContext::MouseTrackingBody::MouseRightHand)
				{
					// choose hold body
					if (selected_body >= 0)
						selected_hold = hitGeomID;
					else
						selected_hold = -1;
				}
				else if (cIndexBody == (int)SimulationContext::MouseTrackingBody::MouseTorsoDir)
				{
					selected_hold = -1;
					int deltax = cX - lX;
					int deltay = cY - lY;
		
					theta_torso -= float (deltax) * 0.5f;
					phi_torso = 0;//(we do not need up-down direction for torso)//-= float (deltay) * 0.5f;

					/*if (mSample.destinationP.size() == 0)
					{
						mSample.sourceP.push_back(mOptCPBP::ControlledPoses::TorsoDir);
						mSample.destinationP.push_back(mTools::getDirectionFromAngles(theta_torso, phi_torso));
					}
					mSample.destinationP[mSample.destinationP.size() - 1] = mTools::getDirectionFromAngles(theta_torso, phi_torso);*/

					lX = cX;
					lY = cY;

					if (showDebugMouseInfo)
						rcPrintString("dx:%d, dy:%d, theta:%f, phi:%f \n",deltax, deltay, theta_torso, phi_torso);
				}
				else
				{
					selected_hold = -1;
				}
			}
			last_hit_geom_id = hitGeomID;
		}
		else
		{
			hitGeomID = odeRaycastGeom(rayBegin.x(), rayBegin.y(), rayBegin.z(),
									   rayDir.x(), rayDir.y(), rayDir.z(), 
									   rayDis, out_pos, out_depth,
									   unsigned long(0x8000), unsigned long(0x8000));
			if (bMouse == 1) // left click
			{
				int deltax = cX - lX;
				int deltay = cY - lY;
				
				int cIndex = mContextRRT->getIndexHold(last_hit_geom_id);
				if (cIndex >= 0)
				{
	//				if (abs(deltax) > 2)
					mContextRRT->holds_body[cIndex].theta -= float (deltax) * 0.5f;
	//				if (abs(deltay) > 2)
					mContextRRT->holds_body[cIndex].phi -= float (deltay) * 0.5f;

					if (mContextRRT->holds_body[cIndex].phi > 90.0f) mContextRRT->holds_body[cIndex].phi = 90.0f;
					if (mContextRRT->holds_body[cIndex].phi < -90.0f) mContextRRT->holds_body[cIndex].phi = -90.0f;

					if (mContextRRT->holds_body[cIndex].theta > 180.0f) mContextRRT->holds_body[cIndex].theta = 180.0f;
					if (mContextRRT->holds_body[cIndex].theta < -180.0f) mContextRRT->holds_body[cIndex].theta = -180.0f;

					Vector3 dir;
				//	Eigen::Matrix3f q1=Eigen::AngleAxisf( mContextRRT->holds_body[cIndex].theta*DEG_TO_RAD,Vector3(0,0,1));


					dir[0] = (float) (cosf (mContextRRT->holds_body[cIndex].theta*DEG_TO_RAD) * cosf (mContextRRT->holds_body[cIndex].phi*DEG_TO_RAD));
					dir[1] = (float) (sinf (mContextRRT->holds_body[cIndex].theta*DEG_TO_RAD) * cosf (mContextRRT->holds_body[cIndex].phi*DEG_TO_RAD));
					dir[2] = (float) (sinf (mContextRRT->holds_body[cIndex].phi*DEG_TO_RAD));
					dir.normalize();

					mContextRRT->holds_body[cIndex].d_ideal = dir;

					if (debugShow)
						rcPrintString("th:%f, ph:%f \n",mContextRRT->holds_body[cIndex].theta, mContextRRT->holds_body[cIndex].phi);
				}
				lX = cX;
				lY = cY;
			}
			else
			{
				last_hit_geom_id = hitGeomID;
				selected_hold = hitGeomID;
			}
		}

		if (bMouse >= 8 && bMouse <= 10)
		{
			if (bMouse == 8)
			{
				disFromCOM -= 0.25f;
			}
			else if (bMouse == 10)
			{
				disFromCOM += 0.25f;
			}
			bMouse = 0;
		}

		if (showDebugMouseInfo)
			rcPrintString("hit body Id:%d \n",hitGeomID);
		return;
	}

	void updateSampleInfo()
	{
		bool updateSample = false;
		if (mSample.destinationP.size() == 0)
			updateSample = true;
		else
		{
			if ((getDirectionFromAngles(theta_torso, phi_torso) - mSample.destinationP[mSample.destinationP.size() - 1]) .norm() > 0.01f)
			{
				updateSample = true;
			}
		}
		if (!mNode::isSetAEqualsSetB(desired_holds_ids, mSample.desired_hold_ids) || updateSample || isBackSpaceHit)
		{
			if (!pauseOptimization)
			{
				mSaveUserDataFunc();
				pauseOptimization = true;
				mSample.isReached = true; // sample is marked by the user as reached
				mSample.isRejected = true; // or sample is marked rejected by the user
			}

			// needs to optimize again
			itr_optimization_for_each_sample = 0;
			mSample.cOptimizationCost = FLT_MAX;
			mSample.numItrFixedCost = 0;
			
			mSample.destinationP.clear();
			mSample.sourceP.clear();
			mSample.dPoint.clear();

			float avgX = 0;
			float maxZ = -FLT_MAX;
			int numDesiredHolds = 0;

			if (!useOfflinePlanning)
				mSample.initial_hold_ids = mOptimizerOnline->startState.hold_bodies_ids;
			else
				mSample.initial_hold_ids = mOptimizerOffline->startState.hold_bodies_ids;

			for (unsigned int i = 0; i < desired_holds_ids.size(); i++)
			{
				if (desired_holds_ids[i] != mSample.desired_hold_ids[i])
				{
					mSample.desired_hold_ids[i] = desired_holds_ids[i];
				}

				if (desired_holds_ids[i] != -1)
				{
					Vector3 dPos = mClimberSampler->getHoldPos(desired_holds_ids[i]);

					mSample.sourceP.push_back((mOptCPBP::ControlledPoses)(mOptCPBP::ControlledPoses::LeftLeg + i));
					mSample.destinationP.push_back(dPos);
					mSample.dPoint.push_back(dPos);

					avgX += dPos.x();
					if (dPos.z() > maxZ)
					{
						maxZ = dPos.z();
					}
					numDesiredHolds++;
				}
			}
			
			if (maxZ >= 0)
			{
				avgX /= (float)(numDesiredHolds + 0.0f);

				Vector3 tPos(avgX, 0, maxZ - boneLength / 4.0f);

				mSample.sourceP.push_back(mOptCPBP::ControlledPoses::MiddleTrunk);
				mSample.destinationP.push_back(tPos);
			}

			// always the last element is the torso direction, we use it later for online changing of the torso direction
			mSample.sourceP.push_back(mOptCPBP::ControlledPoses::TorsoDir);
			mSample.destinationP.push_back(getDirectionFromAngles(theta_torso, phi_torso));

			if (isBackSpaceHit)
			{
				mSaveUserDataFunc();
			}
			mSample.starting_time = timeSpendInPlanning;
			isBackSpaceHit = false;
		}

		return;
	}

	//////////////////////////////////////////// used for online optimization //////////////////////////////////////

	//used for steering in the test mode
	void mSteerFunc(bool isOnlineOptimization)
	{
		mOptimizerOnline->syncMasterContextWithStartState(!isOnlineOptimization);
		
		if (optimizerType == otCMAES && !useOfflinePlanning)
			Debug::throwError("CMAES cannot be used in online mode!");

		if (optimizerType == otCPBP)	
		{
			if (itr_optimization_for_each_sample == 0 && useOfflinePlanning)
				mOptimizerOnline->reset();
			mOptimizerOnline->optimize_the_cost(isOnlineOptimization, mSample.sourceP, mSample.destinationP, mSample.desired_hold_ids, debugShow);
		}

		// check changing of the cost
		float costImprovement = std::max(0.0f,mSample.cOptimizationCost-mOptimizerOnline->current_cost_state);
		rcPrintString("Traj. cost improvement: %f",costImprovement);
		if (costImprovement < noCostImprovementThreshold)
		{
			mSample.numItrFixedCost++;
		}
		else
		{
			mSample.numItrFixedCost = 0;
			
		}
//		if (mOptimizerOnline->current_cost_state < mSample.cOptimizationCost)
		mSample.cOptimizationCost = mOptimizerOnline->current_cost_state;

		//apply the best control to get the start state of next frame
		if (isOnlineOptimization) 
		{
			mStepOptimization(0, true, isOnlineOptimization);
		}

		return;
	}

	void mStepOptimization(int cTimeStep, bool saveIntermediateStates, bool debugPrint = false)
	{
		std::vector<BipedState> nStates;

		mOptimizerOnline->simulateBestTrajectory(saveIntermediateStates, mSample.desired_hold_ids, nStates);// advance_simulation(cTimeStep, debugPrint);
		
		mSample.control_cost += mOptimizerOnline->current_cost_control;

		for (unsigned int i = 0; i < nStates.size(); i++)
		{
			_fromToMotions.push_back(nStates[i]);
		}
	}

}* mTestClimber;

bool advance_time;
bool play_animation;

double total_time_elasped = 0.0f;
mCaseStudy current_case_study = mCaseStudy::movingLimbs;

int cBodyNum;
int cAxisNum;
float dAngle;
bool revertToLastState;

void forwardSimulation()
{
	if (!testClimber)
	{
		myRRTPlanner->mRunPathPlanner(
			useOfflinePlanning ? mRRT::targetMethod::Offiline : mRRT::targetMethod::Online, 
			advance_time, 
			play_animation, 
			(mCaseStudy)(max<int>((int)current_case_study, 0)));
	}
	else
	{
		switch (TestID)
		{
		case TestAngle:
			if (revertToLastState)
			{
				revertToLastState = !revertToLastState;
				mOptimizer->startState = mOptimizer->resetState;
				mOptimizer->syncMasterContextWithStartState(true);
			}
			mContext->setMotorAngle(cBodyNum, cAxisNum, dAngle);
			rcPrintString("axis: %d, angle: %f \n",cAxisNum, dAngle);
			stepOde(timeStep,false);
			break;
		case TestCntroller:
			mTestClimber->runLoopTest(
				advance_time,
				play_animation,
				total_time_elasped);
			break;
		default:
			break;
		}
		
	}
}

void cameraAdjustment(Vector3 _climberCOM)
{
	float disFromCOM = mTestClimber->disFromCOM;

	Vector3 camera_pos = _climberCOM;

	static float theta = -45;
	static float phi = 10;

	if (testClimber)
	{
		mTestClimber->updateCameraPositionByMouse(theta, phi);
	}

	Vector3 cameraDir = mTestControllerClass::getDirectionFromAngles(theta, phi);

	Vector3 cameraLocation = camera_pos + disFromCOM * cameraDir;

	float old_xyz[3] = {xyz[0], xyz[1], xyz[2]};

	xyz[0] = cameraLocation[0];
	xyz[1] = cameraLocation[1];
	xyz[2] = cameraLocation[2];

	bool flag_update_lookAtPos = false;

	if (xyz[1] > -0.25f) // not going through wall
	{
		xyz[1] = -0.25f;
		old_xyz[1] = xyz[1];

		flag_update_lookAtPos = true;
	}
	
	if (xyz[2] < 0.25f) // not going under ground
	{
		xyz[2] = 0.25f;
		old_xyz[2] = xyz[2];

		flag_update_lookAtPos = true;
	}
		
	if (!flag_update_lookAtPos)
	{
		lookAt[0] = _climberCOM[0];
		lookAt[1] = _climberCOM[1];
		lookAt[2] = _climberCOM[2];
	}
	else
	{
		cameraLocation[0] = xyz[0];
		cameraLocation[1] = xyz[1];
		cameraLocation[2] = xyz[2];

		Vector3 lookAtPos = cameraLocation - disFromCOM * cameraDir;

		lookAt[0] = lookAtPos[0];
		lookAt[1] = lookAtPos[1];
		lookAt[2] = lookAtPos[2];
	}
	rcSetViewPoint(xyz[0],xyz[1],xyz[2],lookAt[0],lookAt[1],lookAt[2]);

	mTestClimber->lookDir = cameraDir;
}

////////////////////////////////////////////////////////////// interface functions with drawstuff and unity /////////////////////////////////
void EXPORT_API rcInit()
{
	srand(time(NULL));

	advance_time = true;

	revertToLastState = false;
//	int routeNum = 9;

	mDemoTestClimber hardCodedRoutes = mDemoTestClimber::DemoRoute1;
	/*if (routeNum <=3 && routeNum >= 1)
	{
		hardCodedRoutes = mDemoTestClimber(routeNum);
	}*/

	mContext = new SimulationContext(testClimber, TestID, hardCodedRoutes);
	
	mHoldSampler = new mSampler(mContext);

	BipedState startState;

	startState.hold_bodies_ids.push_back(-1); // 0: ll
	startState.hold_bodies_ids.push_back(-1); // 1: rl
	startState.hold_bodies_ids.push_back(-1); // 2: lh
	startState.hold_bodies_ids.push_back(-1); // 3: rh

	startState.saving_slot_state = mContext->getMasterContextID();

	if (optimizerType == otCPBP)
	{
		mOptimizer = (mController*)(new mOptCPBP(mContext, startState, true));
		mOptimizerOffline = (mController*)(new mOptCPBP(mContext, startState, false));
	}
	else
	{
		mOptimizer = (mController*)(new mOptCMAES(mContext, startState));
	}

	if (!testClimber)
		myRRTPlanner = new mRRT(mContext, mOptimizer, mHoldSampler);
	else
		myRRTPlanner = nullptr;
	
	mTestClimber = new mTestControllerClass(mContext, mOptimizer, mOptimizerOffline, mHoldSampler);

	if (testClimber)
	{
		switch (TestID)
		{
		case mEnumTestCaseClimber::TestAngle:
			cBodyNum = 0;
			cAxisNum = 0;
			dAngle = mContext->getDesMotorAngle(cBodyNum, cAxisNum);
			break;
		default:
			cBodyNum = 0;
			break;
		} 
	}
	else
	{
		cBodyNum = SimulationContext::BodyName::BodyRightArm;
	}

	play_animation = false;

	dAllocateODEDataForThread(dAllocateMaskAll);

	return;
}

void EXPORT_API rcGetClientData(RenderClientData &data)
{
	data.physicsTimeStep = timeStep;
	data.defaultMouseControlsEnabled = false;
	data.maxAllowedTimeStep = timeStep;
}

void EXPORT_API rcUninit()
{
	delete myRRTPlanner;
	delete mTestClimber;
	delete mHoldSampler;
	delete mOptimizer;
	delete mOptimizerOffline;
	delete mContext;

	if (flag_capture_video && testClimber)
	{
		if (fileExists("out.mp4"))
			remove("out.mp4");
		system("screencaps2mp4.bat");
	}

}

static clock_t startingTime = clock();
double passed_time = 0;

void EXPORT_API rcUpdate()
{
	if ((int)(total_time_elasped - passed_time) > 0 && (int)total_time_elasped % 2 == 0)
	{
		startingTime = clock();
		passed_time = total_time_elasped;
	}

	static bool firstRun = true;
	if (firstRun)
	{
		rcSetLightPosition(lightXYZ[0],lightXYZ[1],lightXYZ[2]);
		firstRun = false;
	}

	// reporting total time elapsed
	//rcPrintString("Total CPU time used: %f \n", total_time_elasped);
	rcPrintString("Study Number: %d \n", mContext->_RouteNum + 1);

	startPerfCount();

	if (!pause)
	{
		// forward simulation for reaching desired stances
		forwardSimulation();
	}

	if (!testClimber)
	{
		// simulation part of AI-Climber
		if (myRRTPlanner != nullptr)
			rcPrintString("Total paths found to goal: %d \n", (int)myRRTPlanner->goal_nodes.size());
	}

	/////////////////////////////////////////////////////////// draw stuff ///////////////////////////
	if (testClimber)
	{
		switch (TestID)
		{
		case TestAngle:
			mContext->mDrawStuff(cBodyNum, -1, mContext->masterContext,false,false);
			break;
		case TestCntroller:
			mContext->mDrawStuff(mTestClimber->selected_body, mTestClimber->selected_hold, mContext->masterContext,false,mTestClimber->debugShow);
			break;
		default:
			break;
		}
		
	}
	else
	{
		mContext->mDrawStuff(-1, -1, mContext->masterContext,false,false);
	}

	/////////////////////////////////// Adjust Camera Position ////////////////////////////
	cameraAdjustment(mContext->computeCOM());


	if (myRRTPlanner != nullptr)
	{
		// simulation part of AI-Climber
		if ((int)myRRTPlanner->goal_nodes.size() >= maxNumSimulatedPaths)
		{
			if (!play_animation)
				play_animation = true;

			// simulation part
			rcPrintString("Current animation time:%f \n", myRRTPlanner->cTimeElapsed);
			switch (current_case_study)
			{
			case movingLimbs:
				rcPrintString("Path with min moving limbs among found paths: \n -Path %d with %d moving limbs, \n  and total control cost of %.*lf \n", 
					myRRTPlanner->cGoalPathIndex + 1, 
					(int)myRRTPlanner->cCostGoalMoveLimb, 
					2, 
					myRRTPlanner->cCostGoalControlCost);
				break;
			case Energy:
				rcPrintString("Path with min control cost among found paths: \n -Path %d with %d moving limbs, \n  and total control cost of %.*lf \n"
					, myRRTPlanner->cGoalPathIndex + 1, 
					(int)myRRTPlanner->cCostGoalMoveLimb,
					2, 
					myRRTPlanner->cCostGoalControlCost);
				break;
			default:
				break;
			}

			if (myRRTPlanner->isAnimationFinished)
			{
				if ((int)current_case_study < maxCaseStudySimulation)
					current_case_study = (mCaseStudy)(current_case_study + 1);
			
				myRRTPlanner->isPathFound = false;
				myRRTPlanner->isAnimationFinished = false;
			}

			if ((int)current_case_study >= maxCaseStudySimulation)
			{
				if (flag_capture_video)
				{
					if (fileExists("out.mp4"))
						remove("out.mp4");
					system("screencaps2mp4.bat");
				}
				rcUninit();
				exit(0);
			}
		}
	}

	if (advance_time && !pause)
	{
		auto end = clock();
        auto diff = end-startingTime;
		total_time_elasped = passed_time + (float(diff)/CLOCKS_PER_SEC);//getDurationMs()/1000.0f;

		if (flag_capture_video)
			rcTakeScreenShot();
	}


	return;
}

void EXPORT_API rcOnKeyUp(int key)
{

}

void EXPORT_API rcOnKeyDown(int cmd)
{
	switch (cmd) 
	{
	case 13: // enter
		if (testClimber)
		{
			if (!play_animation)
				mTestClimber->runOptimization();
			/*else
				flag_capture_video = !flag_capture_video;*/
		}
		break;
	case 9: // tab
		/*if (testClimber)
		{
			mTestClimber->debugShow = !mTestClimber->debugShow;
		}*/
		break;
	case 8: //backspace
		if (testClimber)
		{
			switch (TestID)
			{
			case TestAngle:
				revertToLastState = !revertToLastState;
				break;
			case TestCntroller:
				mTestClimber->removeLastSavingState(play_animation);
				break;
			default:
				break;
			}
		}
		break;
	case 'q':	
		rcUninit();
		exit(0);
		break;
	/*case '-':
		myRRTPlanner->cGoalPathIndex--;
		myRRTPlanner->isPathFound = false;
		if (myRRTPlanner->cGoalPathIndex < 0)
		{
			myRRTPlanner->cGoalPathIndex = 0;
		}
		break;
	case '+':
		myRRTPlanner->cGoalPathIndex++;
		myRRTPlanner->isPathFound = false;
		if (myRRTPlanner->cGoalPathIndex >= (int)myRRTPlanner->goal_nodes.size())
		{
			myRRTPlanner->cGoalPathIndex = myRRTPlanner->goal_nodes.size() - 1;
		}
		break;*/
	case ' ':
		play_animation = !play_animation;
		break;
	/*case 'o':
		advance_time = !advance_time;
		break;
	case 'p':
		pause = !pause;
		break;*/
	/*case 'z':
		if (testClimber)
		{
			cBodyNum += 1;
			dAngle = mContext->getDesMotorAngle(cBodyNum, cAxisNum);
		}
		break;
	case 'x':
		if (testClimber)
		{
			cBodyNum -= 1;
			dAngle = mContext->getDesMotorAngle(cBodyNum, cAxisNum);
		}
		break;
	case 'a':
		if (testClimber) 
		{
			cAxisNum += 1;
			dAngle = mContext->getDesMotorAngle(cBodyNum, cAxisNum);
		}
		break;
	case 's':
		if (testClimber)
		{
			cAxisNum -= 1;
			dAngle = mContext->getDesMotorAngle(cBodyNum, cAxisNum);
		}
		break;
	case 'c':
		if (testClimber)
		{
			dAngle += 0.1f;
		}
		break;
	case 'v':
		if (testClimber)
		{
			dAngle -= 0.1f;
		}
		break;*/
	case 'u':
		if (testClimber && TestID == mEnumTestCaseClimber::TestCntroller && mTestClimber->debugShow)
		{
			mTestClimber->update_hold = !mTestClimber->update_hold;
		}
		
		break;
	default:
		break;
	}
	
	if (!mTestClimber->debugShow)
	{
		mTestClimber->update_hold = false;
	}

	return;
}

void EXPORT_API rcOnMouse(float rayStartX, float rayStartY, float rayStartZ, float rayDirX, float rayDirY, float rayDirZ, int button, int x, int y)
{
	Vector3 rBegin(rayStartX,rayStartY,rayStartZ);
	Vector3 rEnd(rayStartX+rayDirX*100.0f,rayStartY+rayDirY*100.0f,rayStartZ+rayDirZ*100.0f);
	if (mTestClimber)
	{
		mTestClimber->rayBegin = rBegin;
		mTestClimber->rayEnd = rEnd;
		mTestClimber->bMouse = button;
		mTestClimber->lX = mTestClimber->cX;
		mTestClimber->lY = mTestClimber->cY;
		mTestClimber->cX = x;
		mTestClimber->cY = y;
	}
}
