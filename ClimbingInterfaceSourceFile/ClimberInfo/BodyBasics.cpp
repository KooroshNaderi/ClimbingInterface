//------------------------------------------------------------------------------
// <copyright file="BodyBasics.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

#include "stdafx.h"
#include <strsafe.h>
#include "resource.h"
#include "BodyBasics.h"
#include <math.h>
#include <vector>
#include <conio.h>

static const float c_JointThickness = 3.0f;
static const float c_TrackedBoneThickness = 6.0f;
static const float c_InferredBoneThickness = 1.0f;
static const float c_HandSize = 30.0f;

/// <summary>
/// Entry point for the application
/// </summary>
/// <param name="hInstance">handle to the application instance</param>
/// <param name="hPrevInstance">always 0</param>
/// <param name="lpCmdLine">command line arguments</param>
/// <param name="nCmdShow">whether to display minimized, maximized, or normally</param>
/// <returns>status</returns>
int APIENTRY wWinMain(    
	_In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPWSTR lpCmdLine,
    _In_ int nShowCmd
)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    CBodyBasics application;
    application.Run(hInstance, nShowCmd);
}

/// <summary>
/// Constructor
/// </summary>
CBodyBasics::CBodyBasics() :
    m_hWnd(NULL),
    m_nStartTime(0),
    m_nLastCounter(0),
    m_nFramesSinceUpdate(0),
    m_fFreq(0),
    m_nNextStatusTime(0LL),
    m_pKinectSensor(NULL),
    m_pCoordinateMapper(NULL),
    m_pBodyFrameReader(NULL),
    m_pD2DFactory(NULL),
    m_pRenderTarget(NULL),
    m_pBrushJointTracked(NULL),
    m_pBrushJointInferred(NULL),
    m_pBrushBoneTracked(NULL),
    m_pBrushBoneInferred(NULL),
    m_pBrushHandClosed(NULL),
    m_pBrushHandOpen(NULL),
    m_pBrushHandLasso(NULL)
{
	cCounterStoreData = 0;
	outHeight = 0;
	disHandToHand = 0;
	counterConvergance = 0;
    LARGE_INTEGER qpf = {0};
    if (QueryPerformanceFrequency(&qpf))
    {
        m_fFreq = double(qpf.QuadPart);
    }
}
  

/// <summary>
/// Destructor
/// </summary>
CBodyBasics::~CBodyBasics()
{
    DiscardDirect2DResources();

    // clean up Direct2D
    SafeRelease(m_pD2DFactory);

    // done with body frame reader
    SafeRelease(m_pBodyFrameReader);

    // done with coordinate mapper
    SafeRelease(m_pCoordinateMapper);

    // close the Kinect Sensor
    if (m_pKinectSensor)
    {
        m_pKinectSensor->Close();
    }

    SafeRelease(m_pKinectSensor);
}

/// <summary>
/// Creates the main window and begins processing
/// </summary>
/// <param name="hInstance">handle to the application instance</param>
/// <param name="nCmdShow">whether to display minimized, maximized, or normally</param>
int CBodyBasics::Run(HINSTANCE hInstance, int nCmdShow)
{
    MSG       msg = {0};
    WNDCLASS  wc;

    // Dialog custom window class
    ZeroMemory(&wc, sizeof(wc));
    wc.style         = CS_HREDRAW | CS_VREDRAW;
    wc.cbWndExtra    = DLGWINDOWEXTRA;
    wc.hCursor       = LoadCursorW(NULL, IDC_ARROW);
    wc.hIcon         = LoadIconW(hInstance, MAKEINTRESOURCE(IDI_APP));
    wc.lpfnWndProc   = DefDlgProcW;
    wc.lpszClassName = L"BodyBasicsAppDlgWndClass";

    if (!RegisterClassW(&wc))
    {
        return 0;
    }

    // Create main application window
    HWND hWndApp = CreateDialogParamW(
        NULL,
        MAKEINTRESOURCE(IDD_APP),
        NULL,
        (DLGPROC)CBodyBasics::MessageRouter, 
        reinterpret_cast<LPARAM>(this));

    // Show window
    ShowWindow(hWndApp, nCmdShow);

    // Main message loop
    while (WM_QUIT != msg.message)
    {
		bool _exit = false;
		if (msg.message == WM_KEYDOWN)
		{
			if ((int)msg.wParam == 27)
			{
				_exit = true;
			}
		}

        Update(_exit);

        while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE))
        {
            // If a dialog message will be taken care of by the dialog proc
            if (hWndApp && IsDialogMessageW(hWndApp, &msg))
            {
                continue;
            }

            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }

    return static_cast<int>(msg.wParam);
}

/// <summary>
/// Main processing function
/// </summary>
void CBodyBasics::Update(bool _exit)
{
    if (!m_pBodyFrameReader)
    {
        return;
    }

    IBodyFrame* pBodyFrame = NULL;

    HRESULT hr = m_pBodyFrameReader->AcquireLatestFrame(&pBodyFrame);

    if (SUCCEEDED(hr))
    {
        INT64 nTime = 0;

        hr = pBodyFrame->get_RelativeTime(&nTime);

        IBody* ppBodies[BODY_COUNT] = {0};

        if (SUCCEEDED(hr))
        {
            hr = pBodyFrame->GetAndRefreshBodyData(_countof(ppBodies), ppBodies);
        }

        if (SUCCEEDED(hr))
        {
            ProcessBody(nTime, BODY_COUNT, ppBodies, pBodyFrame, _exit);
        }

        for (int i = 0; i < _countof(ppBodies); ++i)
        {
            SafeRelease(ppBodies[i]);
        }
    }

    SafeRelease(pBodyFrame);
}

/// <summary>
/// Handles window messages, passes most to the class instance to handle
/// </summary>
/// <param name="hWnd">window message is for</param>
/// <param name="uMsg">message</param>
/// <param name="wParam">message data</param>
/// <param name="lParam">additional message data</param>
/// <returns>result of message processing</returns>
LRESULT CALLBACK CBodyBasics::MessageRouter(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    CBodyBasics* pThis = NULL;
    
    if (WM_INITDIALOG == uMsg)
    {
        pThis = reinterpret_cast<CBodyBasics*>(lParam);
        SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pThis));
    }
    else
    {
        pThis = reinterpret_cast<CBodyBasics*>(::GetWindowLongPtr(hWnd, GWLP_USERDATA));
    }

    if (pThis)
    {
        return pThis->DlgProc(hWnd, uMsg, wParam, lParam);
    }

    return 0;
}

/// <summary>
/// Handle windows messages for the class instance
/// </summary>
/// <param name="hWnd">window message is for</param>
/// <param name="uMsg">message</param>
/// <param name="wParam">message data</param>
/// <param name="lParam">additional message data</param>
/// <returns>result of message processing</returns>
LRESULT CALLBACK CBodyBasics::DlgProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    UNREFERENCED_PARAMETER(wParam);
    UNREFERENCED_PARAMETER(lParam);

    switch (message)
    {
        case WM_INITDIALOG:
        {
            // Bind application window handle
            m_hWnd = hWnd;

            // Init Direct2D
            D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &m_pD2DFactory);

            // Get and initialize the default Kinect sensor
            InitializeDefaultSensor();
        }
        break;

        // If the titlebar X is clicked, destroy app
        case WM_CLOSE:
            DestroyWindow(hWnd);
            break;

        case WM_DESTROY:
            // Quit the main message pump
            PostQuitMessage(0);
            break;
    }

    return FALSE;
}

/// <summary>
/// Initializes the default Kinect sensor
/// </summary>
/// <returns>indicates success or failure</returns>
HRESULT CBodyBasics::InitializeDefaultSensor()
{
    HRESULT hr;

    hr = GetDefaultKinectSensor(&m_pKinectSensor);
    if (FAILED(hr))
    {
        return hr;
    }

    if (m_pKinectSensor)
    {
        // Initialize the Kinect and get coordinate mapper and the body reader
        IBodyFrameSource* pBodyFrameSource = NULL;

        hr = m_pKinectSensor->Open();

        if (SUCCEEDED(hr))
        {
            hr = m_pKinectSensor->get_CoordinateMapper(&m_pCoordinateMapper);
        }

        if (SUCCEEDED(hr))
        {
            hr = m_pKinectSensor->get_BodyFrameSource(&pBodyFrameSource);
        }

        if (SUCCEEDED(hr))
        {
            hr = pBodyFrameSource->OpenReader(&m_pBodyFrameReader);
        }

        SafeRelease(pBodyFrameSource);
    }

    if (!m_pKinectSensor || FAILED(hr))
    {
        SetStatusMessage(L"No ready Kinect found!", 10000, true);
        return E_FAIL;
    }

    return hr;
}

float disBoneBone(const Joint* pJoints, const D2D1_POINT_2F* pJointPoints, JointType joint0, JointType joint1)
{
    TrackingState joint0State = pJoints[joint0].TrackingState;
    TrackingState joint1State = pJoints[joint1].TrackingState;

    // If we can't find either of these joints, exit
    if ((joint0State == TrackingState_NotTracked) || (joint1State == TrackingState_NotTracked))
    {
        return 0.0f;
    }

    // Don't draw if both points are inferred
    if ((joint0State == TrackingState_Inferred) && (joint1State == TrackingState_Inferred))
    {
        return 0.0f;
    }

	float _dis = (float)sqrt(pow(pJointPoints[joint0].x - pJointPoints[joint1].x,(int)2) + pow(pJointPoints[joint0].y - pJointPoints[joint1].y,(int)2));

    // We assume all drawn bones are inferred unless BOTH joints are tracked
    if ((joint0State == TrackingState_Tracked) && (joint1State == TrackingState_Tracked))
    {
        return _dis;
    }
    else
    {
        return _dis;
    }
}

float sumDisBodies(const Joint* pJoints, const D2D1_POINT_2F* pJointPoints)
{
	float sDis = 0.0f;
	sDis += disBoneBone(pJoints, pJointPoints, JointType_Head, JointType_Neck);
    sDis += disBoneBone(pJoints, pJointPoints, JointType_Neck, JointType_SpineShoulder);
    sDis += disBoneBone(pJoints, pJointPoints, JointType_SpineShoulder, JointType_SpineMid);
    sDis += disBoneBone(pJoints, pJointPoints, JointType_SpineMid, JointType_SpineBase);
    sDis += disBoneBone(pJoints, pJointPoints, JointType_SpineShoulder, JointType_ShoulderRight);
    sDis += disBoneBone(pJoints, pJointPoints, JointType_SpineShoulder, JointType_ShoulderLeft);
    sDis += disBoneBone(pJoints, pJointPoints, JointType_SpineBase, JointType_HipRight);
    sDis += disBoneBone(pJoints, pJointPoints, JointType_SpineBase, JointType_HipLeft);
    
    // Right Arm    
    sDis += disBoneBone(pJoints, pJointPoints, JointType_ShoulderRight, JointType_ElbowRight);
    sDis += disBoneBone(pJoints, pJointPoints, JointType_ElbowRight, JointType_WristRight);
    sDis += disBoneBone(pJoints, pJointPoints, JointType_WristRight, JointType_HandRight);
    sDis += disBoneBone(pJoints, pJointPoints, JointType_HandRight, JointType_HandTipRight);
    sDis += disBoneBone(pJoints, pJointPoints, JointType_WristRight, JointType_ThumbRight);

    // Left Arm
    sDis += disBoneBone(pJoints, pJointPoints, JointType_ShoulderLeft, JointType_ElbowLeft);
    sDis += disBoneBone(pJoints, pJointPoints, JointType_ElbowLeft, JointType_WristLeft);
    sDis += disBoneBone(pJoints, pJointPoints, JointType_WristLeft, JointType_HandLeft);
    sDis += disBoneBone(pJoints, pJointPoints, JointType_HandLeft, JointType_HandTipLeft);
    sDis += disBoneBone(pJoints, pJointPoints, JointType_WristLeft, JointType_ThumbLeft);

    // Right Leg
    sDis += disBoneBone(pJoints, pJointPoints, JointType_HipRight, JointType_KneeRight);
    sDis += disBoneBone(pJoints, pJointPoints, JointType_KneeRight, JointType_AnkleRight);
    sDis += disBoneBone(pJoints, pJointPoints, JointType_AnkleRight, JointType_FootRight);

    // Left Leg
    sDis += disBoneBone(pJoints, pJointPoints, JointType_HipLeft, JointType_KneeLeft);
    sDis += disBoneBone(pJoints, pJointPoints, JointType_KneeLeft, JointType_AnkleLeft);
    sDis += disBoneBone(pJoints, pJointPoints, JointType_AnkleLeft, JointType_FootLeft);
	return sDis;
}


float getDisCameraPoints(const CameraSpacePoint& p1, const CameraSpacePoint p2)
{
	float _dis = (float)sqrt(pow(p1.X - p2.X,(int)2) + pow(p1.Y - p2.Y,(int)2) + pow(p1.Z - p2.Z,(int)2));
	return _dis;
}

/// <summary>
/// Handle new body data
/// <param name="nTime">timestamp of frame</param>
/// <param name="nBodyCount">body data count</param>
/// <param name="ppBodies">body data in frame</param>
/// </summary>
void CBodyBasics::ProcessBody(INT64 nTime, int nBodyCount, IBody** ppBodies, IBodyFrame* pBodyFrame, bool _exit)
{
	float _height = 0.0f;
	float _hhdis = 0.0f;
	bool isSatisfied = false;
    if (m_hWnd)
    {
        HRESULT hr = EnsureDirect2DResources();

        if (SUCCEEDED(hr) && m_pRenderTarget && m_pCoordinateMapper)
        {
			std::vector<std::vector<Joint>> bodies_joints(nBodyCount, std::vector<Joint>(JointType_Count));
			std::vector<std::vector<D2D1_POINT_2F>> bodies_jointPoints(nBodyCount, std::vector<D2D1_POINT_2F>(JointType_Count));
			std::vector<HandState> bodies_leftHandState(nBodyCount);
			std::vector<HandState> bodies_rightHandState(nBodyCount);

            m_pRenderTarget->BeginDraw();
            m_pRenderTarget->Clear();

            RECT rct;
            GetClientRect(GetDlgItem(m_hWnd, IDC_VIDEOVIEW), &rct);
            int width = rct.right;
            int height = rct.bottom;

			float maxSizeBody = -1;// maximum body size is the closest
			int index_body = -1;
			for (int i = 0; i < nBodyCount; ++i)
            {
                IBody* pBody = ppBodies[i];
				
                if (pBody)
                {
					BOOLEAN bTracked = false;
                    hr = pBody->get_IsTracked(&bTracked);

                    if (SUCCEEDED(hr) && bTracked)
                    {
                       Joint joints[JointType_Count]; 
                        D2D1_POINT_2F jointPoints[JointType_Count];
                        HandState leftHandState = HandState_Unknown;
                        HandState rightHandState = HandState_Unknown;

                        pBody->get_HandLeftState(&leftHandState);
                        pBody->get_HandRightState(&rightHandState);

						bodies_leftHandState[i] = leftHandState;
						bodies_rightHandState[i] = rightHandState;

                        hr = pBody->GetJoints(_countof(joints), joints);
                        if (SUCCEEDED(hr))
                        {
                            for (int j = 0; j < _countof(joints); ++j)
                            {
                                jointPoints[j] = BodyToScreen(joints[j].Position, width, height);

								bodies_joints[i][j] = joints[j];
								bodies_jointPoints[i][j] = jointPoints[j];
                            }

							float sumDis = sumDisBodies(joints, jointPoints);
							if (sumDis > maxSizeBody)
							{
								maxSizeBody = sumDis;
								index_body = i;
							}
						}
					}
				}
			}

			if (index_body >= 0)
            {
                Joint joints[JointType_Count]; 
                D2D1_POINT_2F jointPoints[JointType_Count];
                HandState leftHandState = bodies_leftHandState[index_body];
                HandState rightHandState = bodies_rightHandState[index_body];

                for (int j = 0; j < _countof(joints); ++j)
                {
                    jointPoints[j] = bodies_jointPoints[index_body][j];
					joints[j] = bodies_joints[index_body][j];
                }

                DrawBody(joints, jointPoints);

                DrawHand(leftHandState, jointPoints[JointType_HandLeft]);
                DrawHand(rightHandState, jointPoints[JointType_HandRight]);

				Vector4 floorClipPlane;
				hr = pBodyFrame->get_FloorClipPlane(&floorClipPlane);

				if(SUCCEEDED(hr))
				{
					float n2 = floorClipPlane.x * floorClipPlane.x + floorClipPlane.y + floorClipPlane.y + floorClipPlane.z * floorClipPlane.z;
					float d = floorClipPlane.w;

					CameraSpacePoint _eFootLeft;
					CameraSpacePoint _eFootRight;
					//CameraSpacePoint _teFootRight;
					if (joints[JointType_AnkleLeft].TrackingState == TrackingState_Tracked)
					{
						float lambda_left = (-d - (floorClipPlane.x * joints[JointType_AnkleLeft].Position.X 
												+ floorClipPlane.y * joints[JointType_AnkleLeft].Position.Y
												+ floorClipPlane.z * joints[JointType_AnkleLeft].Position.Z)) / n2;
						
						_eFootLeft.X = lambda_left * floorClipPlane.x + joints[JointType_AnkleLeft].Position.X;
						_eFootLeft.Y = lambda_left * floorClipPlane.y + joints[JointType_AnkleLeft].Position.Y;
						_eFootLeft.Z = lambda_left * floorClipPlane.z + joints[JointType_AnkleLeft].Position.Z;

						//float a = _eFootLeft.X * floorClipPlane.x;
						//float b = _eFootLeft.Y * floorClipPlane.y;
						//float c = _eFootLeft.Z * floorClipPlane.z;

						//float t1 = a + b + c + floorClipPlane.w;

						D2D1_POINT_2F _showLeftFoot = BodyToScreen(_eFootLeft, width, height);

						m_pRenderTarget->DrawLine(jointPoints[JointType_AnkleLeft], _showLeftFoot, m_pBrushBoneTracked, c_TrackedBoneThickness);
					}
					if (joints[JointType_AnkleRight].TrackingState == TrackingState_Tracked)
					{
						float lambda_right = (-d - (floorClipPlane.x * joints[JointType_AnkleRight].Position.X 
												+ floorClipPlane.y * joints[JointType_AnkleRight].Position.Y
												+ floorClipPlane.z * joints[JointType_AnkleRight].Position.Z)) / n2;
						
						_eFootRight.X = lambda_right * floorClipPlane.x + joints[JointType_AnkleRight].Position.X;
						_eFootRight.Y = lambda_right * floorClipPlane.y + joints[JointType_AnkleRight].Position.Y;
						_eFootRight.Z = lambda_right * floorClipPlane.z + joints[JointType_AnkleRight].Position.Z;
						D2D1_POINT_2F _showRightFoot = BodyToScreen(_eFootRight, width, height);

					//	float t2 = _eFootRight.X * floorClipPlane.x + _eFootRight.Y * floorClipPlane.y + _eFootRight.Z * floorClipPlane.z + floorClipPlane.w;

						m_pRenderTarget->DrawLine(jointPoints[JointType_AnkleRight], _showRightFoot, m_pBrushBoneTracked, c_TrackedBoneThickness);

						
//						a_g = getDisCameraPoints(joints[JointType_AnkleRight].Position, _eFootRight);
//						a_f = getDisCameraPoints(joints[JointType_FootRight].Position, _eFootRight);

					}
//					if (joints[JointType_FootRight].TrackingState == TrackingState_Tracked)
//					{
//						float lambda_right = (-d - (floorClipPlane.x * joints[JointType_FootRight].Position.X 
//												+ floorClipPlane.y * joints[JointType_FootRight].Position.Y
//												+ floorClipPlane.z * joints[JointType_FootRight].Position.Z)) / n2;
//						
//						_teFootRight.X = lambda_right * floorClipPlane.x + joints[JointType_FootRight].Position.X;
//						_teFootRight.Y = lambda_right * floorClipPlane.y + joints[JointType_FootRight].Position.Y;
//						_teFootRight.Z = lambda_right * floorClipPlane.z + joints[JointType_FootRight].Position.Z;
//						D2D1_POINT_2F _showRightFoot = BodyToScreen(_eFootRight, width, height);
//
//					//	float t2 = _eFootRight.X * floorClipPlane.x + _eFootRight.Y * floorClipPlane.y + _eFootRight.Z * floorClipPlane.z + floorClipPlane.w;
//
//						m_pRenderTarget->DrawLine(jointPoints[JointType_FootRight], _showRightFoot, m_pBrushBoneTracked, c_TrackedBoneThickness);
//
////						a_f = getDisCameraPoints(joints[JointType_FootRight].Position, _teFootRight);
//
//					}

					isSatisfied = isConditionForWritingMet(joints, _eFootRight, _eFootLeft, _height, _hhdis);

					for (int i = 0; i < JointType_Count; i++)
						ljoints[i] = joints[i];
					leFootRight = _eFootRight;
					leFootLeft = _eFootLeft;
				}
            }

            hr = m_pRenderTarget->EndDraw();

            // Device lost, need to recreate the render target
            // We'll dispose it now and retry drawing
            if (D2DERR_RECREATE_TARGET == hr)
            {
                hr = S_OK;
                DiscardDirect2DResources();
            }
        }

        if (!m_nStartTime)
        {
            m_nStartTime = nTime;
        }

        double fps = 0.0;

        LARGE_INTEGER qpcNow = {0};
        if (m_fFreq)
        {
            if (QueryPerformanceCounter(&qpcNow))
            {
                if (m_nLastCounter)
                {
                    m_nFramesSinceUpdate++;
                    fps = m_fFreq * m_nFramesSinceUpdate / double(qpcNow.QuadPart - m_nLastCounter);
                }
            }
        }

		if (_exit)
		{
			// write data to a file
			writeToFile(ljoints, leFootRight, leFootLeft);
		}

        WCHAR szStatusMessage[100];
        StringCchPrintf(szStatusMessage, _countof(szStatusMessage), L"ExitingCondition:%d, Height:%.2f, HHDis:%.2f", isSatisfied ? 1 : 0, _height, _hhdis);

        if (SetStatusMessage(szStatusMessage, 1000, false))
        {
            m_nLastCounter = qpcNow.QuadPart;
            m_nFramesSinceUpdate = 0;
        }
    }
}

/// <summary>
/// Set the status bar message
/// </summary>
/// <param name="szMessage">message to display</param>
/// <param name="showTimeMsec">time in milliseconds to ignore future status messages</param>
/// <param name="bForce">force status update</param>
bool CBodyBasics::SetStatusMessage(_In_z_ WCHAR* szMessage, DWORD nShowTimeMsec, bool bForce)
{
    INT64 now = GetTickCount64();

    if (m_hWnd && (bForce || (m_nNextStatusTime <= now)))
    {
        SetDlgItemText(m_hWnd, IDC_STATUS, szMessage);
        m_nNextStatusTime = now + nShowTimeMsec;

        return true;
    }

    return false;
}

/// <summary>
/// Ensure necessary Direct2d resources are created
/// </summary>
/// <returns>S_OK if successful, otherwise an error code</returns>
HRESULT CBodyBasics::EnsureDirect2DResources()
{
    HRESULT hr = S_OK;

    if (m_pD2DFactory && !m_pRenderTarget)
    {
        RECT rc;
        GetWindowRect(GetDlgItem(m_hWnd, IDC_VIDEOVIEW), &rc);  

        int width = rc.right - rc.left;
        int height = rc.bottom - rc.top;
        D2D1_SIZE_U size = D2D1::SizeU(width, height);
        D2D1_RENDER_TARGET_PROPERTIES rtProps = D2D1::RenderTargetProperties();
        rtProps.pixelFormat = D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_IGNORE);
        rtProps.usage = D2D1_RENDER_TARGET_USAGE_GDI_COMPATIBLE;

        // Create a Hwnd render target, in order to render to the window set in initialize
        hr = m_pD2DFactory->CreateHwndRenderTarget(
            rtProps,
            D2D1::HwndRenderTargetProperties(GetDlgItem(m_hWnd, IDC_VIDEOVIEW), size),
            &m_pRenderTarget
        );

        if (FAILED(hr))
        {
            SetStatusMessage(L"Couldn't create Direct2D render target!", 10000, true);
            return hr;
        }

        // light green
        m_pRenderTarget->CreateSolidColorBrush(D2D1::ColorF(0.27f, 0.75f, 0.27f), &m_pBrushJointTracked);

        m_pRenderTarget->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::Yellow, 1.0f), &m_pBrushJointInferred);
        m_pRenderTarget->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::Green, 1.0f), &m_pBrushBoneTracked);
        m_pRenderTarget->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::Gray, 1.0f), &m_pBrushBoneInferred);

        m_pRenderTarget->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::Red, 0.5f), &m_pBrushHandClosed);
        m_pRenderTarget->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::Green, 0.5f), &m_pBrushHandOpen);
        m_pRenderTarget->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::Blue, 0.5f), &m_pBrushHandLasso);
    }

    return hr;
}

/// <summary>
/// Dispose Direct2d resources 
/// </summary>
void CBodyBasics::DiscardDirect2DResources()
{
    SafeRelease(m_pRenderTarget);

    SafeRelease(m_pBrushJointTracked);
    SafeRelease(m_pBrushJointInferred);
    SafeRelease(m_pBrushBoneTracked);
    SafeRelease(m_pBrushBoneInferred);

    SafeRelease(m_pBrushHandClosed);
    SafeRelease(m_pBrushHandOpen);
    SafeRelease(m_pBrushHandLasso);
}

/// <summary>
/// Converts a body point to screen space
/// </summary>
/// <param name="bodyPoint">body point to tranform</param>
/// <param name="width">width (in pixels) of output buffer</param>
/// <param name="height">height (in pixels) of output buffer</param>
/// <returns>point in screen-space</returns>
D2D1_POINT_2F CBodyBasics::BodyToScreen(const CameraSpacePoint& bodyPoint, int width, int height)
{
    // Calculate the body's position on the screen
    DepthSpacePoint depthPoint = {0};
    m_pCoordinateMapper->MapCameraPointToDepthSpace(bodyPoint, &depthPoint);

    float screenPointX = static_cast<float>(depthPoint.X * width) / cDepthWidth;
    float screenPointY = static_cast<float>(depthPoint.Y * height) / cDepthHeight;

    return D2D1::Point2F(screenPointX, screenPointY);
}

/// <summary>
/// Draws a body 
/// </summary>
/// <param name="pJoints">joint data</param>
/// <param name="pJointPoints">joint positions converted to screen space</param>
void CBodyBasics::DrawBody(const Joint* pJoints, const D2D1_POINT_2F* pJointPoints)
{
    // Draw the bones

    // Torso
    DrawBone(pJoints, pJointPoints, JointType_Head, JointType_Neck);
    DrawBone(pJoints, pJointPoints, JointType_Neck, JointType_SpineShoulder);
    DrawBone(pJoints, pJointPoints, JointType_SpineShoulder, JointType_SpineMid);
    DrawBone(pJoints, pJointPoints, JointType_SpineMid, JointType_SpineBase);
    DrawBone(pJoints, pJointPoints, JointType_SpineShoulder, JointType_ShoulderRight);
    DrawBone(pJoints, pJointPoints, JointType_SpineShoulder, JointType_ShoulderLeft);
    DrawBone(pJoints, pJointPoints, JointType_SpineBase, JointType_HipRight);
    DrawBone(pJoints, pJointPoints, JointType_SpineBase, JointType_HipLeft);
    
    // Right Arm    
    DrawBone(pJoints, pJointPoints, JointType_ShoulderRight, JointType_ElbowRight);
    DrawBone(pJoints, pJointPoints, JointType_ElbowRight, JointType_WristRight);
    DrawBone(pJoints, pJointPoints, JointType_WristRight, JointType_HandRight);
    DrawBone(pJoints, pJointPoints, JointType_HandRight, JointType_HandTipRight);
    DrawBone(pJoints, pJointPoints, JointType_WristRight, JointType_ThumbRight);

    // Left Arm
    DrawBone(pJoints, pJointPoints, JointType_ShoulderLeft, JointType_ElbowLeft);
    DrawBone(pJoints, pJointPoints, JointType_ElbowLeft, JointType_WristLeft);
    DrawBone(pJoints, pJointPoints, JointType_WristLeft, JointType_HandLeft);
    DrawBone(pJoints, pJointPoints, JointType_HandLeft, JointType_HandTipLeft);
    DrawBone(pJoints, pJointPoints, JointType_WristLeft, JointType_ThumbLeft);

    // Right Leg
    DrawBone(pJoints, pJointPoints, JointType_HipRight, JointType_KneeRight);
    DrawBone(pJoints, pJointPoints, JointType_KneeRight, JointType_AnkleRight);
    DrawBone(pJoints, pJointPoints, JointType_AnkleRight, JointType_FootRight);

    // Left Leg
    DrawBone(pJoints, pJointPoints, JointType_HipLeft, JointType_KneeLeft);
    DrawBone(pJoints, pJointPoints, JointType_KneeLeft, JointType_AnkleLeft);
    DrawBone(pJoints, pJointPoints, JointType_AnkleLeft, JointType_FootLeft);

    // Draw the joints
    for (int i = 0; i < JointType_Count; ++i)
    {
        D2D1_ELLIPSE ellipse = D2D1::Ellipse(pJointPoints[i], c_JointThickness, c_JointThickness);

        if (pJoints[i].TrackingState == TrackingState_Inferred)
        {
            m_pRenderTarget->FillEllipse(ellipse, m_pBrushJointInferred);
        }
        else if (pJoints[i].TrackingState == TrackingState_Tracked)
        {
            m_pRenderTarget->FillEllipse(ellipse, m_pBrushJointTracked);
        }
    }
}

void writePosOnFile(FILE* _mFile, const CameraSpacePoint& _iPos)
{
	fprintf_s(_mFile,"%f,%f,%f\n",_iPos.X,_iPos.Y,_iPos.Z);
	return;
}

bool CBodyBasics::isConditionForWritingMet(const Joint* pJoints, const CameraSpacePoint& _rFoot, const CameraSpacePoint& _lFoot, float& _height, float& _hhDis)
{
	bool isSatisfied = true;
	for (int i = 0; i < JointType_Count && isSatisfied; i++)
	{
		if (pJoints[i].TrackingState == TrackingState_NotTracked || pJoints[i].TrackingState == TrackingState_Inferred)
        {
			isSatisfied = false;
		}
	}

	cCounterStoreData++;
	if (cCounterStoreData < 200)
		isSatisfied = false;// throw data away

	
	float cHeight = 0.0f;
	cHeight += (getDisCameraPoints(pJoints[JointType_AnkleRight].Position, _rFoot)
				+ getDisCameraPoints(pJoints[JointType_AnkleLeft].Position, _lFoot)) / 2.0f;
	cHeight += (getDisCameraPoints(pJoints[JointType_KneeRight].Position, pJoints[JointType_AnkleRight].Position)
							+ getDisCameraPoints(pJoints[JointType_KneeLeft].Position, pJoints[JointType_AnkleLeft].Position)) / 2.0f;
	cHeight += (getDisCameraPoints(pJoints[JointType_HipRight].Position, pJoints[JointType_KneeRight].Position)
							+ getDisCameraPoints(pJoints[JointType_HipLeft].Position, pJoints[JointType_KneeLeft].Position)) / 2.0f;
	float midPosY = (pJoints[JointType_HipRight].Position.Y + pJoints[JointType_HipLeft].Position.Y) / 2.0f;
	cHeight += pJoints[JointType_SpineBase].Position.Y - midPosY;
	cHeight += getDisCameraPoints(pJoints[JointType_SpineMid].Position, pJoints[JointType_SpineBase].Position);
	cHeight += getDisCameraPoints(pJoints[JointType_SpineShoulder].Position, pJoints[JointType_SpineMid].Position);
	cHeight += getDisCameraPoints(pJoints[JointType_SpineShoulder].Position, pJoints[JointType_Neck].Position);
	cHeight += 2.0f * getDisCameraPoints(pJoints[JointType_Head].Position, pJoints[JointType_Neck].Position);

	_height = cHeight;

	if (fabs(outHeight - cHeight) > 0.01f)
	{
		outHeight = cHeight;
		counterConvergance = 0;
		isSatisfied = false; // removing convergance issue
	}

	float cDisHandToHand = 0.0f;
	cDisHandToHand += (getDisCameraPoints(pJoints[JointType_ShoulderRight].Position, pJoints[JointType_ElbowRight].Position)
							+ getDisCameraPoints(pJoints[JointType_ShoulderLeft].Position, pJoints[JointType_ElbowLeft].Position)) / 2.0f;
	cDisHandToHand += (getDisCameraPoints(pJoints[JointType_ElbowRight].Position, pJoints[JointType_WristRight].Position)
							+ getDisCameraPoints(pJoints[JointType_ElbowLeft].Position, pJoints[JointType_WristLeft].Position)) / 2.0f;
	cDisHandToHand += (getDisCameraPoints(pJoints[JointType_WristRight].Position, pJoints[JointType_HandTipRight].Position)
							+ getDisCameraPoints(pJoints[JointType_WristLeft].Position, pJoints[JointType_HandTipLeft].Position)) / 2.0f;
	cDisHandToHand += (getDisCameraPoints(pJoints[JointType_SpineShoulder].Position, pJoints[JointType_ShoulderRight].Position)
							+ getDisCameraPoints(pJoints[JointType_SpineShoulder].Position, pJoints[JointType_ShoulderLeft].Position)) / 2.0f;
	
	_hhDis = cDisHandToHand;
	if (fabs(disHandToHand - cDisHandToHand) > 0.01f)
	{
		disHandToHand = cDisHandToHand;
		counterConvergance = 0;
		isSatisfied = false; // removing convergance issue
	}

	counterConvergance++;
	if (counterConvergance < 10)
	{
		isSatisfied = false;
	}

	return isSatisfied;
}

// write in a file the correct formatting of bones to be used in another program
bool CBodyBasics::writeToFile(const Joint* pJoints, const CameraSpacePoint& _rFoot, const CameraSpacePoint& _lFoot)
{
	FILE* _mFile = fopen("mClimberReadKinect.txt", "w+");

	fprintf_s(_mFile,"#X,Y,Z\n");

	// write torso positions
	writePosOnFile(_mFile, pJoints[JointType_Head].Position);
	writePosOnFile(_mFile, pJoints[JointType_Neck].Position);
	writePosOnFile(_mFile, pJoints[JointType_SpineShoulder].Position);
	writePosOnFile(_mFile, pJoints[JointType_SpineMid].Position);
	writePosOnFile(_mFile, pJoints[JointType_SpineBase].Position);
	writePosOnFile(_mFile, pJoints[JointType_ShoulderRight].Position);
	writePosOnFile(_mFile, pJoints[JointType_ShoulderLeft].Position);
	writePosOnFile(_mFile, pJoints[JointType_HipRight].Position);
	writePosOnFile(_mFile, pJoints[JointType_HipLeft].Position);

	// write arm positions
	writePosOnFile(_mFile, pJoints[JointType_ElbowRight].Position);
	writePosOnFile(_mFile, pJoints[JointType_ElbowLeft].Position);
	writePosOnFile(_mFile, pJoints[JointType_WristRight].Position);
	writePosOnFile(_mFile, pJoints[JointType_WristLeft].Position);
	writePosOnFile(_mFile, pJoints[JointType_HandTipRight].Position);
	writePosOnFile(_mFile, pJoints[JointType_HandTipLeft].Position);

	// write leg positions
	writePosOnFile(_mFile, pJoints[JointType_KneeRight].Position);
	writePosOnFile(_mFile, pJoints[JointType_KneeLeft].Position);
	writePosOnFile(_mFile, pJoints[JointType_AnkleRight].Position);
	writePosOnFile(_mFile, pJoints[JointType_AnkleLeft].Position);
	writePosOnFile(_mFile, pJoints[JointType_FootRight].Position);
	writePosOnFile(_mFile, pJoints[JointType_FootLeft].Position);

	writePosOnFile(_mFile, _rFoot);
	writePosOnFile(_mFile, _lFoot);

	fclose(_mFile);

	exit(0);
	return true;
}

/// <summary>
/// Draws one bone of a body (joint to joint)
/// </summary>
/// <param name="pJoints">joint data</param>
/// <param name="pJointPoints">joint positions converted to screen space</param>
/// <param name="pJointPoints">joint positions converted to screen space</param>
/// <param name="joint0">one joint of the bone to draw</param>
/// <param name="joint1">other joint of the bone to draw</param>
void CBodyBasics::DrawBone(const Joint* pJoints, const D2D1_POINT_2F* pJointPoints, JointType joint0, JointType joint1)
{
    TrackingState joint0State = pJoints[joint0].TrackingState;
    TrackingState joint1State = pJoints[joint1].TrackingState;

    // If we can't find either of these joints, exit
    if ((joint0State == TrackingState_NotTracked) || (joint1State == TrackingState_NotTracked))
    {
        return;
    }

    // Don't draw if both points are inferred
    if ((joint0State == TrackingState_Inferred) && (joint1State == TrackingState_Inferred))
    {
        return;
    }

    // We assume all drawn bones are inferred unless BOTH joints are tracked
    if ((joint0State == TrackingState_Tracked) && (joint1State == TrackingState_Tracked))
    {
        m_pRenderTarget->DrawLine(pJointPoints[joint0], pJointPoints[joint1], m_pBrushBoneTracked, c_TrackedBoneThickness);
    }
    else
    {
        m_pRenderTarget->DrawLine(pJointPoints[joint0], pJointPoints[joint1], m_pBrushBoneInferred, c_InferredBoneThickness);
    }
}

/// <summary>
/// Draws a hand symbol if the hand is tracked: red circle = closed, green circle = opened; blue circle = lasso
/// </summary>
/// <param name="handState">state of the hand</param>
/// <param name="handPosition">position of the hand</param>
void CBodyBasics::DrawHand(HandState handState, const D2D1_POINT_2F& handPosition)
{
    D2D1_ELLIPSE ellipse = D2D1::Ellipse(handPosition, c_HandSize, c_HandSize);

    switch (handState)
    {
        case HandState_Closed:
            m_pRenderTarget->FillEllipse(ellipse, m_pBrushHandClosed);
            break;

        case HandState_Open:
            m_pRenderTarget->FillEllipse(ellipse, m_pBrushHandOpen);
            break;

        case HandState_Lasso:
            m_pRenderTarget->FillEllipse(ellipse, m_pBrushHandLasso);
            break;
    }
}
