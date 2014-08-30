#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "windows.h"
#include <time.h>

#include "GL/glee.h"

#include "consts.h"

#include "nanorod.h"
#include "global.h"
#include "film.h"
#include "tracer.h"
#include "obj_object.h"
#include "texture.h"

#include "IL/ilut.h"
#include "GL/glut.h"
#include "GL/glui.h"

#include "gpu_util.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "gpu_util.cu"
#include "tracer.cu"

#define _MSVC
#include "cWorldVol.h"
#include "cCameraVol.h"
#include "chai3d/src/chai3d.h"

///////////////////////////

float *deviceData = NULL;
int *idData = NULL;

///////////////////////////
//

//***Haptic globals***
float *hostData = NULL;
int *hostIdData = NULL;

void initHaptic();
// function called before exiting the application
void closeHaptic(void);
// main graphics callback
void updateHapticGraphics(void);
// main haptics loop
void updateHaptics(void);

const int MAX_DEVICES           = 1;
// a world that contains all objects of the virtual environment
cWorldVol* world;
// a camera that renders the world in a window display
cCameraVol* camera;
// a light source to illuminate the objects in the virtual scene
cLight *light;
// a little "chai3d" bitmap logo at the bottom of the screen
cBitmap* logo;
// width and height of the current window display
int displayW;
int displayH;
// a haptic device handler
cHapticDeviceHandler* handler;
// a table containing pointers to all haptic devices detected on this computer
cGenericHapticDevice* hapticDevices[MAX_DEVICES];
// a table containing pointers to label which display the position of
// each haptic device
cLabel* labels[MAX_DEVICES];
cGenericObject* rootLabels;
// number of haptic devices detected
int numHapticDevices;
// table containing a list of 3D cursors for each haptic device
cShapeSphere* cursors[MAX_DEVICES];
// table containing a list of lines to display velocity
cShapeLine* velocityVectors[MAX_DEVICES];
// material properties used to render the color of the cursors
cMaterial matCursorButtonON;
cMaterial matCursorButtonOFF;
// status of the main simulation haptics loop
bool simulationRunning;
// root resource path
string resourceRoot;
// damping mode ON/OFF
bool useDamping;
// force field mode ON/OFF
bool useForceField;
// has exited haptics simulation thread
bool simulationFinished;
//Camera tool vector
vect3d vCameraToolVect;
//Toool position
cVector3d posTool;
// a virtual tool representing the haptic device in the scene
cGeneric3dofPointer* tool;
// a spherical object representing the volume so it can have material properties 
// and we can change them when volume values change
cShapeSphere* object0;
double stiffnessMax;
double forceMax;
double dampingMax;
int typeOfForce = 0; 
int shouldDrawAxes = 1;
ObjObject *pCap0;

//***Haptic Globals end***

//***Haptic Functions***
void initHaptic()
{
	displayW  = 0;
	displayH  = 0;
	numHapticDevices = 0;
	simulationRunning = false;
	useDamping = false;
	useForceField = true;
	simulationFinished = false;
    // create a new world.
    world = new cWorldVol();

    // set the background color of the environment
    // the color is defined by its (R,G,B) components.
    world->setBackgroundColor(0.0, 0.0, 0.0);

    // create a camera and insert it into the virtual world
	camera = new cCameraVol(world);
	world->addChild(camera);

    // position and oriente the camera
    camera->set( cVector3d (0.5, 0.0, 0.0),    // camera position (eye)
                 cVector3d (0.0, 0.0, 0.0),    // lookat position (target)
                 cVector3d (0.0, 0.0, 1.0));   // direction of the "up" vector

    // set the near and far clipping planes of the camera
    // anything in front/behind these clipping planes will not be rendered
    camera->setClippingPlanes(0.01, 10.0);

    // create a light source and attach it to the camera
    light = new cLight(world);
    camera->addChild(light);                   // attach light to camera
    light->setEnabled(true);                   // enable light source
    light->setPos(cVector3d( 2.0, 0.5, 1.0));  // position the light source
    light->setDir(cVector3d(-2.0, 0.5, 1.0));  // define the direction of the light beam
    //-----------------------------------------------------------------------
    // HAPTIC DEVICES / TOOLS
    //-----------------------------------------------------------------------

    // create a haptic device handler
    handler = new cHapticDeviceHandler();

    // read the number of haptic devices currently connected to the computer
    numHapticDevices = handler->getNumDevices();

    // limit the number of devices to MAX_DEVICES
    numHapticDevices = cMin(numHapticDevices, MAX_DEVICES);

    // create a node on which we will attach small labels that display the
    // position of each haptic device
    rootLabels = new cGenericObject();
    camera->m_front_2Dscene.addChild(rootLabels);

    // create a small label as title
    cLabel* titleLabel = new cLabel();
    rootLabels->addChild(titleLabel);

    // define its position, color and string message
    titleLabel->setPos(0, 30, 0);
    titleLabel->m_fontColor.set(1.0, 1.0, 1.0);
    titleLabel->m_string = "Haptic Device Pos [mm]:";

    // for each available haptic device, create a 3D cursor
    // and a small line to show velocity
    int i = 0;
    while (i < numHapticDevices)
    {
        // get a handle to the next haptic device
        cGenericHapticDevice* newHapticDevice;
        handler->getDevice(newHapticDevice, i);

        // open connection to haptic device
        newHapticDevice->open();

		// initialize haptic device
		newHapticDevice->initialize();

        // store the handle in the haptic device table
        hapticDevices[i] = newHapticDevice;

        // retrieve information about the current haptic device
        cHapticDeviceInfo info = newHapticDevice->getSpecifications();

		// create a 3D tool and add it to the world
		tool = new cGeneric3dofPointer(world);
		world->addChild(tool);

		// connect the haptic device to the tool
		tool->setHapticDevice(hapticDevices[i]);

		// initialize tool by connecting to haptic device
		tool->start();

		// map the physical workspace of the haptic device to a larger virtual workspace.
		tool->setWorkspaceRadius(1.0);

		// define a radius for the tool
		tool->setRadius(0.03);

		// read the scale factor between the physical workspace of the haptic
		// device and the virtual workspace defined for the tool
		double workspaceScaleFactor = tool->getWorkspaceScaleFactor();

		// define a maximum stiffness that can be handled by the current
		// haptic device. The value is scaled to take into account the
		// workspace scale factor
		stiffnessMax = info.m_maxForceStiffness / workspaceScaleFactor;
		forceMax = info.m_maxForce;

		// define the maximum damping factor that can be handled by the
		// current haptic device. The The value is scaled to take into account the
		// workspace scale factor
		dampingMax = info.m_maxLinearDamping / workspaceScaleFactor; 

		/////////////////////////////////////////////////////////////////////////
		// OBJECT 0: "VIBRATIONS"
		////////////////////////////////////////////////////////////////////////

		// temp variable
		cGenericEffect* newEffect;

		// create a sphere and define its radius
		object0 = new cShapeSphere(2.0);

		// add object to world
		world->addChild(object0);

		// set the position of the object at the center of the world
		object0->setPos(0.0, 0.0, 0.0);

		object0->setUseTexture(false);

		// create a haptic viscous effect
		newEffect = new cEffectVibration(object0);
		object0->addEffect(newEffect);

		newEffect = new cEffectSurface(object0);
		object0->addEffect(newEffect);

		//newEffect = new cEffectViscosity(object0);
		//object0->addEffect(newEffect);

		//newEffect = new cEffectMagnet(object0);
		//object0->addEffect(newEffect);


        // create a cursor by setting its radius
        cShapeSphere* newCursor = new cShapeSphere(0.01);

        // add cursor to the world
        world->addChild(newCursor);

        // add cursor to the cursor table
        cursors[i] = newCursor;

        // create a small line to illustrate velocity
        cShapeLine* newLine = new cShapeLine(cVector3d(0,0,0), cVector3d(0,0,0));
        velocityVectors[i] = newLine;

        // add line to the world
        world->addChild(newLine);

        // create a string that concatenates the device number and model name.
        string strID;
        cStr(strID, i);
        string strDevice = "#" + strID + " - " +info.m_modelName;

        // attach a small label next to the cursor to indicate device information
        cLabel* newLabel = new cLabel();
        newCursor->addChild(newLabel);
        newLabel->m_string = strDevice;
        newLabel->setPos(0.00, 0.02, 0.00);
        newLabel->m_fontColor.set(1.0, 1.0, 1.0);

        // if the device provided orientation sensing (stylus), a reference
        // frame is displayed
        if (info.m_sensedRotation == true)
        {
            // display a reference frame
            newCursor->setShowFrame(true);

            // set the size of the reference frame
            newCursor->setFrameSize(0.05, 0.05);
        }

        // crate a small label to indicate the position of the device
        cLabel* newPosLabel = new cLabel();
        rootLabels->addChild(newPosLabel);
        newPosLabel->setPos(0, -20 * i, 0);
        newPosLabel->m_fontColor.set(0.6, 0.6, 0.6);
        labels[i] = newPosLabel;

        // increment counter
        i++;
    }

	// simulation in now running
    simulationRunning = true;

    // create a thread which starts the main haptics rendering loop
    cThread* hapticsThread = new cThread();
    hapticsThread->set(updateHaptics, CHAI_THREAD_PRIORITY_HAPTICS);
}

void closeHaptic(void)
{
    // stop the simulation
    simulationRunning = false;

    // wait for graphics and haptics loops to terminate
    while (!simulationFinished) { cSleepMs(100); }

    // close all haptic devices
    int i=0;
    while (i < numHapticDevices)
    {
        hapticDevices[i]->close();
        i++;
    }
}

float getElecCellValue(int x, int y, int z, float *elecData, int *idData)
{
	if( x < 0 || x >= VOL_X || 
		y < 0 || y >= VOL_Y ||
		z < 0 || z >= VOL_Z )	// Hard-code it for now
	{
		return 0;
	}

	unsigned offset = x + y * VOL_X + z * VOL_X * VOL_Y;

	return *(elecData + offset);
}

cVector3d toolCoord2VolCoord(cVector3d toolCoord)
{
	//TODO: now it's rotation dependent, it shouldn't
	cVector3d result;
	
	float hapticWorkSpaceRadius = 1.f;
	if(numHapticDevices > 0)
		hapticWorkSpaceRadius = hapticDevices[0]->getSpecifications().m_workspaceRadius;
	result.x = toolCoord.y / (hapticWorkSpaceRadius * 2.f) * VOL_X;
	result.y = toolCoord.x / (hapticWorkSpaceRadius * 2.f) * VOL_Y;
	result.z = toolCoord.z / (hapticWorkSpaceRadius * 2.f) * VOL_Z;
/* %%%Before
	cVector3d result;
	result.x = (toolCoord.y * VOL_X) / 0.4f;
	result.y = (-toolCoord.x * VOL_Y) / 0.4f;
	result.z = (toolCoord.z * VOL_Z) / 0.4f;
*/
	return result;
}

void setCameraToolVector()
{
	if(numHapticDevices > 0)
	{
		//Save previous camera before doing anything
		scene.setPreviousCameraCenter(scene.getCamera());
		//Get vector: camera center -> tool
		hapticDevices[0]->getPosition(posTool);
		posTool = toolCoord2VolCoord(posTool);
		vect3d convertedPosTool(posTool.x, posTool.y, posTool.z);
		//Vector: camera center -> tool
		vCameraToolVect = convertedPosTool - *scene.getPreviousCameraCenter();
	}
}

//Use the camera tool vector to set the new position of the tool object in GPU so it does not move
//when transforms are made to the scene
void setToolPositionGPU()
{
	if(numHapticDevices > 0)
	{
		//TODO: for now, just rotation
		//Transform cameraToolVect with the same transformations as the ones for the camera
		vect3d posNewCameraTool;
		vect3d vtmp;
		mat_rot(vCameraToolVect, vtmp);
		//Add this vector to the camera center point
		vect3d *posCamera = scene.computeCameraCenter(scene.getCamera());
		posNewCameraTool = *posCamera + vtmp;
		//Generate vector:  cameraToolVect -> posNewCameraTool
		vect3d vOldToolNewTool = posNewCameraTool - (vCameraToolVect + *scene.getPreviousCameraCenter());
		//Translate the object in the GPU 
		setObjectCenterGPU(vOldToolNewTool, 1);
	}
}

//Just detect the tool's position and pass it to the GPU
void moveToolPositionGPU()
{
	if(numHapticDevices > 0)
	{
		//Transform cameraToolVect with the same transformations as the ones for the camera
		static cVector3d previousPosTool(0,0,0);
		cVector3d translation = posTool - previousPosTool;
		vect3d convertedTranslation(translation.x, translation.y, translation.z);
		//Translate the object in the GPU 
		translateObjectGPU(convertedTranslation, 1);
		previousPosTool = posTool;
	}
}

//Just detect the tool's position and translate coordinates to volume coordinates
void moveToolPositionCPU()
{
	if(numHapticDevices > 0)
	{
		//Transform cameraToolVect with the same transformations as the ones for the camera
		cVector3d newPosTool;
		//Get vector: camera center -> tool
		hapticDevices[0]->getPosition(newPosTool);
		//Account for the tool's actual imprecisions
		//%%%newPosTool.mul(5);
		newPosTool = toolCoord2VolCoord(newPosTool);
		posTool = newPosTool;
	}
}

void updateHapticGraphics(void)
{
	// update content of position label
    // read position of device an convert into millimeters
    if(numHapticDevices > 0)
	{
		//This is for drawing from the volume's camera
		moveToolPositionGPU();

		//This is for drawing from the haptic's camera
		cVector3d pos;
		hapticDevices[0]->getPosition(pos);
		pos.mul(5);

		// create a string that concatenates the device number and its position.
		string strID;
		cStr(strID, 0);
		string strLabel = "#" + strID + "  x: ";

		cStr(strLabel, pos.x, 2);
		strLabel = strLabel + "   y: ";
		cStr(strLabel, pos.y, 2);
		strLabel = strLabel + "  z: ";
		cStr(strLabel, pos.z, 2);

		labels[0]->m_string = strLabel;
	}

//TODO: need to draw it correctly
//	camera->renderView(displayW, displayH);

    // check for any OpenGL errors
    GLenum err;
    err = glGetError();
    if (err != GL_NO_ERROR) printf("Error:  %s\n", gluErrorString(err));
}

void ResetForces()
{
	//Vibration
	object0->m_material.setVibrationFrequency(0);
	object0->m_material.setVibrationAmplitude(0);
	//Friction
	object0->m_material.setStiffness(0);
	object0->m_material.setStaticFriction(0);
	object0->m_material.setViscosity(0);
}

void updateHaptics(void)
{
    // main haptic simulation loop
    while(simulationRunning)
    {
 
		if(numHapticDevices > 0)
		{
			// read position of haptic device
			cVector3d newPosition;
			hapticDevices[0]->getPosition(newPosition);

			// read orientation of haptic device
			cMatrix3d newRotation;
			hapticDevices[0]->getRotation(newRotation);

			// update position and orientation of cursor
			cursors[0]->setPos(newPosition);
			cursors[0]->setRot(newRotation);

			// read linear velocity from device
			cVector3d linearVelocity;
			hapticDevices[0]->getLinearVelocity(linearVelocity);

			// update arrow
			velocityVectors[0]->m_pointA = newPosition;
			velocityVectors[0]->m_pointB = cAdd(newPosition, linearVelocity);

			// read user button status
			bool buttonStatus;
			hapticDevices[0]->getUserSwitch(0, buttonStatus);

			// adjustthe  color of the cursor according to the status of
			// the user switch (ON = TRUE / OFF = FALSE)
			if (buttonStatus)
			{
				cursors[0]->m_material = matCursorButtonON;
			}
			else
			{
				cursors[0]->m_material = matCursorButtonOFF;
			}

			//get value from data at the position of the tool (the converted position)
			moveToolPositionCPU();
			float dataRange = fEnd-fStart;
			cVector3d newForce (0,0,0);
			float val = getElecCellValue(posTool.x + VOL_X/2.f, posTool.y + VOL_Y/2.f, posTool.z+VOL_Z/2.f, hostData, hostIdData) / dataRange;
			//printf("%f\n",val);	

			// set haptic properties according to the voxel inside the volume
			// NOTE that there are two ways this is being done, first, object0
			// has some properties, then some forces will be applied through the
			// tool variable and also some forces are applied directly to the 
			// haptic device through hapticDevices[0]->setForce()
			
			if(typeOfForce == 1)
			{
				ResetForces();	
				//Vibration
				object0->m_material.setVibrationFrequency(50.f);
				object0->m_material.setVibrationAmplitude(1.0 * forceMax * val);
			}
			//Magnetic Force
			//object0->m_material.setStiffness(0.1 * stiffnessMax * val);
			//object0->m_material.setMagnetMaxForce(0.1 * 2000.0 * val);
			//object0->m_material.setMagnetMaxDistance(0.05);
			//object0->m_material.setViscosity(1.0 * dampingMax);
			else if(typeOfForce == 2)
			{
				ResetForces();	
				//Friction
				object0->m_material.setStiffness(0.1 * stiffnessMax * val);
				object0->m_material.setDynamicFriction(1.0 * 2000.0 * val);
				object0->m_material.setViscosity(1.0 * dampingMax);
			}
			else if (typeOfForce == 3)
			{
				ResetForces();	
				//Vibration
				object0->m_material.setVibrationFrequency(50.f);
				object0->m_material.setVibrationAmplitude(1.0 * forceMax * val);
				//Friction
				object0->m_material.setStiffness(0.1 * stiffnessMax * val);
				object0->m_material.setStaticFriction(1.0 * 2000.0 * val);
				object0->m_material.setViscosity(1.0 * dampingMax);
			}

/*Question: which one reveals more the high value areas? Which one reveals more the structure of the rod?*/

			// apply force field
			if (typeOfForce == 0)
			{
				//Compute force
				double Kp = 2000.0 * val; // [N/m]
				cVector3d force = cMul(-Kp, newPosition);
				newForce.add(force);
				//Damp
				cHapticDeviceInfo info = hapticDevices[0]->getSpecifications();
				double Kv = info.m_maxLinearDamping*val;
				cVector3d force2 = cMul(-Kv, linearVelocity);
				newForce.add(force2);
			}
	    
			// compute global reference frames for each object
			world->computeGlobalPositions(true);

			// 4 position and orientation of tool
			tool->updatePose();

			// compute interaction forces
			tool->computeInteractionForces();

		
			if(typeOfForce == 0)
			{
				// send computed force to haptic device (direct forces)
				hapticDevices[0]->setForce(newForce);
			}
			else
			{
				// send forces to device (like vibration)
				tool->applyForces();
			}
		}

    }
    
    // exit haptics thread
    simulationFinished = true;
}




//***Haptic Functions End***

//Shader
//shader variables
GLuint  fragShader;
GLuint  vertShader;
GLuint	program;
GLint	fragCompiled;
GLint	vertCompiled;

const char *vertProgram;
const char *fragProgram;

void setupShaders()
{

	if (!GL_ARB_vertex_program) 
	{ 
		
		printf("No shaders!");
		return;		 
	}

	FILE *file;

	file = fopen("./blend.frag","r");

	if(file==NULL)
	{
		MessageBox(NULL,"Couldn't open frag file.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		exit(0);
	}

	char *fragProg;
	int size=0;

	fseek(file, 0, SEEK_END);
	size = ftell(file)+1;
	fragProg = new char[size];
	fseek(file, 0, SEEK_SET);
	size = fread(fragProg,1,size,file);
	fragProg[size]='\0';
	fclose(file);
	fragProgram = fragProg;

	file = fopen("blend.vert","r");

	if(file==NULL)
	{
		MessageBox(NULL,"Couldn't open vert file.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		exit(0);
	}

	char *vertProg;
	size=0;

	fseek(file, 0, SEEK_END);
	size = ftell(file)+1;
	vertProg = new char[size];
	fseek(file, 0, SEEK_SET);
	size = fread(vertProg,1,size,file);
	vertProg[size]='\0';
	fclose(file);
	vertProgram = vertProg;

	vertShader = glCreateShader(GL_VERTEX_SHADER);
	fragShader = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(fragShader, 1, &fragProgram, NULL);
	glShaderSource(vertShader, 1, &vertProgram, NULL);

	glCompileShader(vertShader);
//	getErrors();
//	printShaderInfoLog(vertShader);
	glGetShaderiv(vertShader, GL_COMPILE_STATUS, &vertCompiled);

	glCompileShader(fragShader);
//	getErrors();
//	printShaderInfoLog(fragShader);
	glGetShaderiv(fragShader, GL_COMPILE_STATUS, &fragCompiled);

	program = glCreateProgram();
	glAttachShader(program, vertShader);
	glAttachShader(program, fragShader);

	glLinkProgram(program);
	glUseProgram(program);
}

void drawAxes()
{
	glDisable(GL_TEXTURE_2D);
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(-100.f, 100.f, -100.f, 100.f, -10000.f, 10000.f);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glLineWidth(2);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glPushMatrix();
	glRotatef(2.f, 1, 1, 0);
	glBegin(GL_LINES);
	{
		glColor4f(1.f, 0, 0, 0.2f);
		glVertex3f(-100.f, 0,0);
		glVertex3f(100.f,0,0);
		glColor4f(0, 1.f, 0, 0.2f);
		glVertex3f(0, -100.f,0);
		glVertex3f(0,100.f,0);
		glColor4f(0, 0, 1.f, 0);
		glVertex3f(0, 0,-1000.f);
		glColor4f(0, 0, 1.f, 1.f);
		glVertex3f(0,0,1000.f);
	}
	glEnd();
	glPopMatrix();
	glLineWidth(1);
	glDisable(GL_BLEND);
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glEnable(GL_TEXTURE_2D);
	//	check error
	GLenum err = glGetError();
	if(err != GL_NO_ERROR)
	{
		printf("[GL ERROR] %s - %d : 0x%x\n", __FILE__, __LINE__, err);
	}

}

void drawCap()
{
	glDisable(GL_TEXTURE_2D);
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(-1000.f, 1000.f, -1000.f, 1000.f, -10000.f, 10000.f);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glLineWidth(2);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glPushMatrix();
	glBegin(GL_TRIANGLES);
	{
		for(int i =0; i < pCap0->getTriCount(); i++)
		{
			glVertex3f(
				pCap0->getTriangle(i)->_vertices->data[0], 
				pCap0->getTriangle(i)->_vertices->data[1], 
				pCap0->getTriangle(i)->_vertices->data[2]); 
		}
	}
	glEnd();
	glPopMatrix();
	glLineWidth(1);
	glDisable(GL_BLEND);
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glEnable(GL_TEXTURE_2D);
	//	check error
	GLenum err = glGetError();
	if(err != GL_NO_ERROR)
	{
		printf("[GL ERROR] %s - %d : 0x%x\n", __FILE__, __LINE__, err);
	}
}

//

static
void resize(int w, int h)
{
	//%%%
	displayW = w;
	displayH = h;
	//%%%
    glViewport(0, 0, w, h);	
	//%%%
    // update position of labels
    rootLabels->setPos(10, displayH-70, 0);	
	//%%%
}

static 
void destroy()
{	
	if(deviceData)
	{
		cudaFree(deviceData);
		deviceData = NULL;
	}

	if(idData)
	{
		cudaFree(idData);
		idData = NULL;
	}

	if(deviceTexData)
	{
		cudaFree(deviceTexData);
	}

	nanoGeoDestroy();
	nanoPlaneDestroy();
	internalCap0Destroy();
	internalCap1Destroy();
	SliceDestroy();
	global_destroy();

	exit(EXIT_SUCCESS);
}

int iWinId;
void idle() 
{
	glutSetWindow(iWinId);
	glutPostRedisplay();
}

clock_t nTick = 0;
GLUI_RadioGroup *pCMGroup = NULL;
GLUI_EditText *pImgPath = NULL;

//	Zooming
static int InitViewZ = 6100;
const int MaxViewZ = 12000;
const int MinViewZ = 100;

static int nCurrViewZ = InitViewZ;
static int nZoomStep = -10;

static int nRotStep = 1;
void zoom_cam(Camera *pCam, float deltaStep);
void rotate_cam(Camera *pCam, float deltaAngle, vect3d &axis);
void capture();

static int volCount = 0;

static
void display()
{
	nTick = clock();

	glClear(GL_COLOR_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, 1, 0, 1);

	//%%%
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(
		0, 0, 1, 
		0, 0, 0,  
		0, 1, 0); 
    glDisable(GL_DEPTH_TEST);
	//%%%

	scene.setTFMode(iTfMode);
	scene.compute();
	scene.render( pCMGroup->get_int_val() == 0 ? NULL : pImgPath->get_text());

	//%%%
	//Render Haptic graphcis
	updateHapticGraphics();
	//%%%
	if(shouldDrawAxes) drawAxes();

	glutSwapBuffers();

	clock_t nCount = clock() - nTick;
	printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b FPS: %.4f", 1000.f / (nCount * 1.0));

	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess)
	{
		printf("DUDE: %s \n", cudaGetErrorString(err));
	}

	//	
	//if(volCount <= 360)
	//{
	//	capture();
	//	//nCurrViewZ += nZoomStep;
	//	//zoom_cam(scene.getCamera(), -nZoomStep);
	//	rotate_cam(scene.getCamera(), -nRotStep, vect3d(0, 0.3, 1));
	//}

}



///		Rotation
static void rotate_cam(Camera *pCam, float deltaAngle, vect3d &axis)
{
	PerpCamera *pPCam = dynamic_cast<PerpCamera *>(pCam);

	//	Rotate
	deltaAngle *= - PIon180;

	set_matrix(sinf(deltaAngle), cosf(deltaAngle), axis);

	vect3d tmp;
	mat_rot(pPCam->_eyePos, tmp);	vecCopy(pPCam->_eyePos, tmp);
	mat_rot(pPCam->_ctrPos, tmp);	vecCopy(pPCam->_ctrPos, tmp);
	mat_rot(pPCam->_upVec, tmp);	vecCopy(pPCam->_upVec, tmp);
	mat_rot(pPCam->_rightVec, tmp);	vecCopy(pPCam->_rightVec, tmp);
	mat_rot(pPCam->_dir, tmp);	vecCopy(pPCam->_dir, tmp);
}

static void zoom_cam(Camera *pCam, float deltaStep)
{
	PerpCamera *pPCam = dynamic_cast<PerpCamera *>(pCam);

	vect3d deltaVec;
	vect3d eye(pPCam->_eyePos[0], pPCam->_eyePos[1], pPCam->_eyePos[2]);
	vect3d ctr(pPCam->_ctrPos[0], pPCam->_ctrPos[1], pPCam->_ctrPos[2]); 
	vect3d viewVec;
	points2vec(eye, ctr, viewVec);
	vecCopy(deltaVec, viewVec); normalize(deltaVec); vecScale(deltaVec, deltaStep, deltaVec);
	//printf("->%.3f,%.3f,%.3f \n", viewVec.data[0], viewVec.data[0], viewVec.data[0]);

	point2point(pPCam->_eyePos, deltaVec, pPCam->_eyePos);
}

static 
void specialKey(int key, int x, int y)
{
	switch(key)
	{
	///
	///		Zooming
	///
	case GLUT_KEY_UP:
		//if( (nCurrViewZ - nZoomStep) >=  MinViewZ)
		{
			nCurrViewZ -= nZoomStep;
			zoom_cam(scene.getCamera(), nZoomStep);
			printf("Zoom-in to %d\n", nCurrViewZ);
		}
		break;

	case GLUT_KEY_DOWN:
		//if( (nCurrViewZ + nZoomStep) <=  MaxViewZ)
		{
			nCurrViewZ += nZoomStep;
			zoom_cam(scene.getCamera(), -nZoomStep);
			printf("Zoom-out to %d\n", nCurrViewZ);
		}
		break;

	///
	///		Rotation
	///
	case GLUT_KEY_RIGHT:
		printf("  Right Rot: %d\n", nRotStep);
		rotate_cam(scene.getCamera(), -nRotStep, vect3d(0, 0.3, 1));
		break;

	case GLUT_KEY_LEFT:
		printf("  Left Rot: %d\n", -nRotStep);
		rotate_cam(scene.getCamera(), nRotStep, vect3d(0, 0.3, 1));
		break;
	}
}

void capture()
{
	//	Taking & Saving the screenshot				   
	if(ilutGLScreen())
	{					  
	  ilEnable(IL_FILE_OVERWRITE);
	  char path[20] = {0};
	  sprintf(path, "Y:/tony/vol_%d.jpg", volCount ++);
	  if(ilSaveImage(path))
	  {
		 printf("Screenshot saved successfully as \'%s\'!\n", path);
	  }
	  else
	  {
		 printf("Sorry, DevIL cannot save your screenshot...\n");
	  }
	}
	else
	{
	  printf(" Sorry man, DevIL screenshot taking failed...\n");
	}
}

static 
void key(unsigned char key, int x, int y)
{		
	PerpCamera *pCam = (PerpCamera*)scene.getCamera();
    switch (key) 
    {
	case 'm':
		printf("DIR: %.5f. %.5f. %.5f\n", pCam->_dir[0], pCam->_dir[1], pCam->_dir[2]);
		printf("CTR: %.5f. %.5f. %.5f\n", pCam->_eyePos[0], pCam->_eyePos[1], pCam->_eyePos[2]);
		printf("UP : %.5f. %.5f. %.5f\n", pCam->_upVec[0], pCam->_upVec[1], pCam->_upVec[2]);
		break;

	case 'c':
	case 'C':
		capture();
		break;

    case 27 : 
    case 'q':
        destroy();
        break;
    }

    //glutPostRedisplay();
}

void printUsage()
{
	char *strUsage =	"{ How to Use } \n\n"
						" Wheel Up\\Down to Zoom-in\\out \n"
						" Mouse Drag to Rotate \n"
						" C: save image \n"
						" Q: quit \n\n";
	printf(strUsage);
}

///		Radio Group Callback
GLUI_RadioGroup *pGroup = NULL;
GLUI_Panel *pHPPal = NULL;

void file_callback(int pParam)
{
	printf("Image Path: %s\n", pImgPath->get_text());
	loadTexture(pImgPath->get_text());
}

GLUI_Panel *pClrP = NULL;
GLUI_Panel *pClrChooseP = NULL;
GLUI_Panel *pFileChooseP = NULL;

void color_map_choice_callback(int pParam)
{
	int val = pCMGroup->get_int_val();

	switch(val)
	{
	case 0:	//	Values
		pClrP->enable();
		pClrChooseP->enable();
		pFileChooseP->disable();
		break;

	case 1: // Picture
		pClrP->disable();
		pClrChooseP->disable();
		pFileChooseP->enable();
#ifndef DATA_2D
		pFileChooseP->disable();	
#endif		
		loadTexture(pImgPath->get_text());

		break;
	}

	mMode = val;
}

void radio_group_callback(int pParam)
{
	int val = pGroup->get_int_val();

	switch(val)
	{
	case 0:		
		pHPPal->disable();
		printf("\nAverage Mode Selected.\n");
		break;

	case 1:
		pHPPal->disable();
		printf("\nSolid Mode Selected.\n");
		break;

	case 2:	//	hermite
		pHPPal->enable();
		printf("\nHermite Mode Selected.\n");
		break;

	case 3:
		pHPPal->disable();
		printf("\nFirst Mode Selected.\n");
		break;
	}

	iTfMode = val;
}

static int pressX = -1;
static int pressY = -1;

void myGlutMouse(int button, int button_state, int x, int y)
{
	//	Zooming
	//
	if(button==GLUT_WHEEL_DOWN) 
	{
		if( (nCurrViewZ + nZoomStep) <=  MaxViewZ)
		{
			nCurrViewZ += nZoomStep;
			zoom_cam(scene.getCamera(), -nZoomStep);
			printf("Zoom-out to %d\n", nCurrViewZ);
		}
	}
	if(button==GLUT_WHEEL_UP) 
	{
		if( (nCurrViewZ - nZoomStep) >=  MinViewZ)
		{
			nCurrViewZ -= nZoomStep;
			zoom_cam(scene.getCamera(), nZoomStep);
			printf("Zoom-in to %d\n", nCurrViewZ);
		}
	}

	//	Rotating
	//
	if (button==GLUT_LEFT_BUTTON && button_state==GLUT_DOWN) 
	{
		pressX = x;
		pressY = y;
		//%%%
		//generate a vector from camera to tool
//		setCameraToolVector();
		//%%%
	}
	if (button==GLUT_LEFT_BUTTON && button_state==GLUT_UP) 
	{
		int dx = x - pressX;
		int dy = pressY - y;
		if( (abs(dx) + abs(dy)) > 3 )
		{
			float fSpeed = 0.2;
			Camera *pCam = scene.getCamera();

			//	Drag Vec
			vect3d dragVec;
			vect3d tmpHori, tmpVert;
			vecScale(pCam->_rightVec, dx, tmpHori);
			vecScale(pCam->_upVec, dy, tmpVert);
			point2point(dragVec, tmpHori, dragVec);
			point2point(dragVec, tmpVert, dragVec);

			float dragLen = sqrt( (float)dx * dx + (float)dy * dy );

			vect3d axis(0, 1, 0);
			cross_product(dragVec, pCam->_dir, axis);
			normalize(axis);

			rotate_cam(pCam, fSpeed * dragLen, axis);
			//%%%
			//Assigns new tool position
//			setToolPositionGPU();
			//%%%
		}
	}
}

///
#include "data_loader.cu"

///
///		
///
int main(int argc, char* argv[])
{

	cudaGLSetGLDevice(0);

	printUsage();

	//	Window Setup
	glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);	
	
	glutInitWindowSize(WinWidth, WinHeight);
    glutInitWindowPosition(WinLeft, WinTop);
    iWinId = glutCreateWindow(WinTitle);
    
	glutReshapeFunc(resize);
    glutDisplayFunc(display);
	glutIdleFunc(idle);
    glutKeyboardFunc(key);
	glutSpecialFunc(specialKey);
	
	
	//
	scene.init();
	global_init();

	atexit(destroy);

#if 1

	///
	///		Load Files and copy data to GPU
	///

	const unsigned x_dim = VOL_X;
	const unsigned y_dim = VOL_Y;
	const unsigned z_dim = VOL_Z;

	cudaMalloc(&deviceData, sizeof(float) * 3 * x_dim * y_dim * z_dim);	//	x, y, z
	if(deviceData == NULL)
	{
		printf("CUDA Memory Allocation failed...\n");
		system("pause");
	}

	cudaMalloc(&idData, sizeof(int) * x_dim * y_dim * z_dim);	//	x, y, z
	if(idData == NULL)
	{
		printf("CUDA Memory Allocation failed...\n");
		system("pause");
	}

	//%%%Initialize host data for haptic use 
	hostData = (float*)malloc(sizeof(float) * 3 * x_dim * y_dim * z_dim);
	hostIdData = (int*)malloc(sizeof(int) * x_dim * y_dim * z_dim);
	//%%%

#ifndef DATA_2D
	bool bLoaded = loadData(	x_dim, y_dim, z_dim, 
								DATA_PATH, 
								deviceData, idData,
								hostData, hostIdData);	
#else
	bool bLoaded = loadData2D(	x_dim, y_dim, z_dim, 
								DATA_PATH, 
								deviceData,
								hostData);	
#endif
	scene.setElecData(deviceData, idData);

#endif

	//	GLUI
	GLUI *glui = GLUI_Master.create_glui( "Param Control", 0, WinWidth + WinLeft, WinTop );
	
	//	Kernel Part
	//
	{
		GLUI_Panel *pPal0 = glui->add_panel("Nanorod Param");
		// Multi-sample Count
		GLUI_Spinner *pMS = glui->add_spinner_to_panel(pPal0, "Anti-Alias #", GLUI_SPINNER_INT, &nMultiSampleCount);
		pMS->set_int_limits(1, 40);
		pMS->set_speed(1);
#ifndef DATA_2D
		pMS->disable();
#endif
		//	Sampling Dist Radius
		GLUI_Spinner *pDF = glui->add_spinner_to_panel(pPal0, "Sampling Rad #", GLUI_SPINNER_FLOAT, &fSamplingDeltaFactor);
		pDF->set_float_limits(0.01, 60);
		pDF->set_speed(1);
#ifndef DATA_2D
		pDF->disable();
#endif
		//	Show Geo
		GLUI_Checkbox *pGeoChk = glui->add_checkbox_to_panel(pPal0, "Show Nanorod", &bShowGeo);
		GLUI_Checkbox *pRodChk = glui->add_checkbox_to_panel(pPal0, "Data in Nanorod", &bOnlyInRod);
//%%%	Type of force to be rendered
		GLUI_Listbox *pListForce = glui->add_listbox_to_panel(pPal0, "Type of Force", &typeOfForce);
		{
			pListForce->add_item(0, "Force 1");
			pListForce->add_item(1, "Force 2");
			pListForce->add_item(2, "Force 3");
			pListForce->add_item(3, "Force 2 & 3");
		}
		GLUI_Checkbox *drawAxesChk = glui->add_checkbox_to_panel(pPal0, "Draw axes", &shouldDrawAxes);
//%%%	
		GLUI_Spinner *pNanoAlpha = glui->add_spinner_to_panel(pPal0, "Nanorod Alpha", GLUI_SPINNER_FLOAT, &fNanoAlpha);
		pNanoAlpha->set_float_limits(0, 1);
		GLUI_Checkbox *pSliceChk = glui->add_checkbox_to_panel(pPal0, "Show Slice", &bShowSlice);
		GLUI_Checkbox *pPlaneChk = glui->add_checkbox_to_panel(pPal0, "Show Plane", &bShowPlane);
		GLUI_Spinner *pPlaneAlpha = glui->add_spinner_to_panel(pPal0, "Plane Alpha", GLUI_SPINNER_FLOAT, &fPlaneAlpha);
		pNanoAlpha->set_float_limits(0, 1);

		GLUI_Panel *pTFPal = glui->add_panel("Transfer Function");
#ifndef DATA_2D
		pTFPal->disable();
#endif
		pGroup = glui->add_radiogroup_to_panel(pTFPal, NULL, -1, radio_group_callback);
		glui->add_radiobutton_to_group( pGroup, "Average" );
		glui->add_radiobutton_to_group( pGroup, "Solid" );
		glui->add_radiobutton_to_group( pGroup, "Hermite" );

		pHPPal = glui->add_panel("Hermite Param");
		glui->add_edittext_to_panel(pHPPal, "P0 Val", GLUI_EDITTEXT_FLOAT, &fP0_val);
		glui->add_edittext_to_panel(pHPPal, "P0 Deriv", GLUI_EDITTEXT_FLOAT, &fP0_der);
		glui->add_edittext_to_panel(pHPPal, "P1 Val", GLUI_EDITTEXT_FLOAT, &fP1_val);
		glui->add_edittext_to_panel(pHPPal, "P1 Deriv", GLUI_EDITTEXT_FLOAT, &fP1_der);
		pHPPal->disable();
	}

	{
		GLUI *glui2 = GLUI_Master.create_glui( "", 0, WinWidth + WinLeft + 200, WinTop);
		
		//
		GLUI_Panel *pPalP = glui2->add_panel("Clip Plane");		
		glui2->add_checkbox_to_panel(pPalP, "Enable", &bClipPlaneEnabled);
		
		glui2->add_statictext_to_panel(pPalP, "Plane Center");
		GLUI_Spinner *pc0 = glui2->add_spinner_to_panel(pPalP, "", GLUI_SPINNER_FLOAT, planeCtr + 0);	
		pc0->set_float_limits(-(VOL_X/2), (VOL_X/2));
		GLUI_Spinner *pc1 = glui2->add_spinner_to_panel(pPalP, "", GLUI_SPINNER_FLOAT, planeCtr + 1);
		pc1->set_float_limits(-(VOL_Y/2), (VOL_Y/2));
		GLUI_Spinner *pc2 = glui2->add_spinner_to_panel(pPalP, "", GLUI_SPINNER_FLOAT, planeCtr + 2);
		pc2->set_float_limits(-(VOL_Z/2), (VOL_Z/2));

		glui2->add_statictext_to_panel(pPalP, "Plane Normal");
		GLUI_Spinner *pn0 = glui2->add_spinner_to_panel(pPalP, "", GLUI_SPINNER_FLOAT, planeNorm + 0);
		GLUI_Spinner *pn1 = glui2->add_spinner_to_panel(pPalP, "", GLUI_SPINNER_FLOAT, planeNorm + 1);
		GLUI_Spinner *pn2 = glui2->add_spinner_to_panel(pPalP, "", GLUI_SPINNER_FLOAT, planeNorm + 2);

		//
		GLUI_Panel *pSelP = glui2->add_panel("Data Selection");		
#ifndef DATA_2D
		pSelP->disable();
#endif
		glui2->add_checkbox_to_panel(pSelP, "ID 1", mark + 0);
		glui2->add_checkbox_to_panel(pSelP, "ID 2", mark + 1);
		glui2->add_checkbox_to_panel(pSelP, "ID 3", mark + 2);
		glui2->add_checkbox_to_panel(pSelP, "ID 4", mark + 3);
		
	}
	
	{
		GLUI *glui3 = GLUI_Master.create_glui( "", 0, WinWidth + WinLeft + 400, WinTop);
		
		GLUI_Panel *pChoiceP = glui3->add_panel("Choice");
#ifndef DATA_2D
		pChoiceP->disable();	
#endif		
		pCMGroup = glui->add_radiogroup_to_panel(pChoiceP, NULL, -1, color_map_choice_callback);
		glui3->add_radiobutton_to_group( pCMGroup, "Values");
		glui3->add_radiobutton_to_group( pCMGroup, "Picture");
		pCMGroup->set_int_val(mMode);

		glui3->add_separator();

		pClrP = glui3->add_panel("Color Map Values");
	
		GLUI_Spinner *pcl0 = glui3->add_spinner_to_panel(pClrP, "Val-0", GLUI_SPINNER_FLOAT, knotValues + 0);	
		pcl0->set_float_limits(-20, 20);
		GLUI_Spinner *pcl1 = glui3->add_spinner_to_panel(pClrP, "Val-1", GLUI_SPINNER_FLOAT, knotValues + 1);	
		pcl1->set_float_limits(-20, 20);
		GLUI_Spinner *pcl2 = glui3->add_spinner_to_panel(pClrP, "Val-2", GLUI_SPINNER_FLOAT, knotValues + 2);	
		pcl2->set_float_limits(-20, 20);
		GLUI_Spinner *pcl3 = glui3->add_spinner_to_panel(pClrP, "Val-3", GLUI_SPINNER_FLOAT, knotValues + 3);	
		pcl3->set_float_limits(-20, 20);
		GLUI_Spinner *pcl4 = glui3->add_spinner_to_panel(pClrP, "Val-4", GLUI_SPINNER_FLOAT, knotValues + 4);	
		pcl4->set_float_limits(-20, 20);

		//
		pClrChooseP = glui3->add_panel("Color Map Colors");
#ifndef DATA_2D
		pClrChooseP->disable();	
#endif	
		GLUI_Listbox *pList0 = glui3->add_listbox_to_panel(pClrChooseP, "Color 0", knotColors + 0);
		{
			pList0->add_item(0, "White");
			pList0->add_item(1, "Black");
			pList0->add_item(2, "Red");
			pList0->add_item(3, "Orange");
			pList0->add_item(4, "Yellow");
			pList0->add_item(5, "Green");
			pList0->add_item(6, "Cyan");
			pList0->add_item(7, "Blue");
			pList0->add_item(8, "Purple");
			pList0->add_item(9, "Gray");
		}
		pList0->set_int_val(knotColors[0]);

		GLUI_Listbox *pList1 = glui3->add_listbox_to_panel(pClrChooseP, "Color 1", knotColors + 1);
		{
			pList1->add_item(0, "White");
			pList1->add_item(1, "Black");
			pList1->add_item(2, "Red");
			pList1->add_item(3, "Orange");
			pList1->add_item(4, "Yellow");
			pList1->add_item(5, "Green");
			pList1->add_item(6, "Cyan");
			pList1->add_item(7, "Blue");
			pList1->add_item(8, "Purple");
			pList1->add_item(9, "Gray");
		}
		pList1->set_int_val(knotColors[1]);

		GLUI_Listbox *pList2 = glui3->add_listbox_to_panel(pClrChooseP, "Color 2", knotColors + 2);
		{
			pList2->add_item(0, "White");
			pList2->add_item(1, "Black");
			pList2->add_item(2, "Red");
			pList2->add_item(3, "Orange");
			pList2->add_item(4, "Yellow");
			pList2->add_item(5, "Green");
			pList2->add_item(6, "Cyan");
			pList2->add_item(7, "Blue");
			pList2->add_item(8, "Purple");
			pList2->add_item(9, "Gray");
		}
		pList2->set_int_val(knotColors[2]);

		GLUI_Listbox *pList3 = glui3->add_listbox_to_panel(pClrChooseP, "Color 3", knotColors + 3);
		{
			pList3->add_item(0, "White");
			pList3->add_item(1, "Black");
			pList3->add_item(2, "Red");
			pList3->add_item(3, "Orange");
			pList3->add_item(4, "Yellow");
			pList3->add_item(5, "Green");
			pList3->add_item(6, "Cyan");
			pList3->add_item(7, "Blue");
			pList3->add_item(8, "Purple");
			pList3->add_item(9, "Gray");
		}
		pList3->set_int_val(knotColors[3]);

		GLUI_Listbox *pList4 = glui3->add_listbox_to_panel(pClrChooseP, "Color 4", knotColors + 4);	
		{
			pList4->add_item(0, "White");
			pList4->add_item(1, "Black");
			pList4->add_item(2, "Red");
			pList4->add_item(3, "Orange");
			pList4->add_item(4, "Yellow");
			pList4->add_item(5, "Green");
			pList4->add_item(6, "Cyan");
			pList4->add_item(7, "Blue");
			pList4->add_item(8, "Purple");
			pList4->add_item(9, "Gray");
		}
		pList4->set_int_val(knotColors[4]);

		glui3->add_separator();

		pFileChooseP = glui3->add_panel("File Choose");
		pImgPath = glui3->add_edittext_to_panel(pFileChooseP, "Img Path", GLUI_EDITTEXT_TEXT, NULL, -1, file_callback);
		pImgPath->set_text(CM_IMG);
		glui3->add_spinner_to_panel(pFileChooseP, "Start", GLUI_SPINNER_FLOAT, &fStart);	
		glui3->add_spinner_to_panel(pFileChooseP, "End", GLUI_SPINNER_FLOAT, &fEnd);	
#ifndef DATA_2D
		pFileChooseP->disable();	
#endif		
	}
	GLUI_Master.set_glutIdleFunc(idle);
	GLUI_Master.set_glutMouseFunc(myGlutMouse);
	
	///
	///		Setup Scene
	///

	{
		///		Camera
		///
		CamType	eCamType = PERSP;
		SamplingType eSplType = STRATIFIED;

		////	Top
		//vect3d ctr(0, 0, InitViewZ); 
		//vect3d view(0, 0, -1);
		//vect3d up(0, 1, 0);

		////	Bottom
		//vect3d ctr(0, 0, -InitViewZ); 
		//vect3d view(0, 0, 1);
		//vect3d up(0, 1, 0);

		////	Side
		//vect3d ctr(InitViewZ, 0, 0); 
		//vect3d view(-1, 0, 0);
		//vect3d up(0, 0, 1);

		////	90
		//vect3d ctr(0, InitViewZ, 0); 
		//vect3d view(0, -1, 0);
		//vect3d up(0, 0, 1);
		
		////	look-down-cap
		//vect3d ctr(0, InitViewZ, InitViewZ * 0.3 - 20); 
		//vect3d view(0, -1, -0.3);
		//vect3d up(0, 0.3, 1);

		////	cut-plane
		//vect3d ctr(5539.27002, 4021.99023, -929.16742); 
		//vect3d view(-121.07777, -87.91293, 20.30981);
		//vect3d up(0.53435, 0.43612, 5.07333);

		////	Poly-Plane
		// vect3d view(-194.79440, 156.42287, -24.24892);
		// vect3d ctr(5516.48730, -4429.82227, 686.71802);
		// vect3d up(-0.36831, 0.33132, 5.09597);

		//	Poly-Piece
		//
		 vect3d view(15.01775, 179.30638, 89.58064);
		 vect3d ctr(-531.08990, -6341.02490, -3167.94580);
		 vect3d up(-0.18784, -2.27411, 4.58339);

		////	SQW - view
		//vect3d view(13.06205, -26.43243, -9.57651);
		//vect3d ctr(-2612.41113, 5286.48486, 1915.30151);
		//vect3d up(0.66073, -1.43767, 4.86939);

		//	Set Camera
		Camera *pCam = NULL;
		switch(eCamType)
		{
		case PERSP:
			pCam = new PerpCamera(300, ctr, up, view, 500, ViewPlaneRatio);
			break;

		case ORTHO:
			pCam = new OrthoCamera(ctr, up, view, 10, ViewPlaneRatio);
			break;
		}
		pCam->setSampler(eSplType);
		pCam->setMultiSamplingCount(nMultiSampleCount);
		scene.setCamera(pCam);
		scene.setAmbiColor(vect3d(0,0,0));

		///		Generate Rays for GPU	
		///
		sendConstants2GPU();

		///		Volume Cube
		///
		vect3d cubeCtr(0, 0, 0);
		vect3d vertVec(0, 1, 0);
		vect3d horiVec(0, 0, 1);

		Cube *pCube0 = new Cube(x_dim, z_dim, y_dim, cubeCtr, vertVec, horiVec);
		scene.addObject(pCube0);
#ifndef DATA_2D
		//%%%Add my little cube representing the tool
		Cube *pCubeTool = new Cube(x_dim / 30.f, z_dim / 46.8f, y_dim / 30.f, cubeCtr, vertVec, horiVec);
		scene.addObject(pCubeTool);
		scene.setDataDim(x_dim, y_dim, z_dim);
#endif
		int half_x = x_dim / 2;
		int half_y = y_dim / 2;
		int half_z = z_dim / 2;

		Tracer::setVolBBox( - half_x - 1, half_x + 1,
							- half_y - 1, half_y + 1,
							- half_z - 1, half_z + 1);

		copySceneGeomotry();

#if 1	//c10w20
		float factorNano = 0.90;//1
		float factorSlice = factorNano;
		float factorCap0 = 0.63;//0.9
		float factorCap1 = 0.5;
		float factorPlane = factorNano;
		float offset = 0; //50; //43;//34.5;

		//float factorNano = 1;//1
		//float factorSlice = factorNano;
		//float factorCap0 = factorNano;//0.9
		//float factorCap1 = factorNano;
		//float factorPlane = factorNano;
		//float offset = 34.5;
#else

		float factorNano = 1;
		float factorSlice = 1;
		float factorCap0 = 1;
		float factorCap1 = 1;
		float factorPlane = 1;
		float offset = 0;
#endif
		///		
		///		Nanorod Geometry
		///
		printf("- Loading Nanorod Geometry ...");
		{
			ObjObject *pObj = new ObjObject(0, 0, 0, 0);	
			pObj->load("nanorod.obj");
			pObj->setSmooth(false);
			
			vect3d spec(1,1,1);
			vect3d diff(1,1,1);
			vect3d ambi(1,1,1);
			pObj->setMaterial(spec, diff, ambi, 70);

			vect3d axis(1, 0, 0);
			float angle = -90;
			pObj->rotate(angle, axis);
			pObj->scale(0.82 * factorNano, 0.82 * factorNano, 0.70 * factorNano);		
			pObj->translate(0,0,-3 - offset);

			printf("Done \n");

			printf("- Transfering Nanorod Geometry to GPU...");
			copyNanoGeo(pObj, offset);
			printf("Done \n");
		}

		printf("- Loading Slice Geometry ...");
		{
			ObjObject *pObj = new ObjObject(0, 0, 0, 0);	
			pObj->load("slice.obj");
			pObj->setSmooth(false);
			
			vect3d spec(1,1,1);
			vect3d diff(1,1,1);
			vect3d ambi(0.2,0.2,0.2);
			pObj->setMaterial(spec, diff, ambi, 70);

			vect3d axis(1, 0, 0);
			float angle = -90;
			pObj->rotate(angle, axis);

			vect3d axis0(0, 0, 1);
			float angle0 = -120;
			pObj->rotate(angle0, axis0);

			pObj->scale(0.82 * factorSlice, 0.82 * factorSlice, 0.70 * factorSlice);		
			pObj->translate(0,0,-3 - offset);

			printf("Done \n");

			printf("- Transfering Slice Geometry to GPU...");
			copySlice(pObj);
			printf("Done \n");
		}

		///		1. Internal Cap 0
		{
			ObjObject *pObj = new ObjObject(0, 0, 0, 0);	
			pCap0 = pObj;
			pObj->load("nanorod.obj");
			pObj->setSmooth(false);
			
			vect3d spec(0.3,0.3,0.3);
			vect3d diff(0.3, 0.3, 0.3);
			vect3d ambi(0.3, 0.3, 0.3);
			pObj->setMaterial(spec, diff, ambi, 70);

			vect3d axis(1, 0, 0);
			float angle = -90;
			pObj->rotate(angle, axis);
			pObj->scale(0.82 * factorCap0, 0.82 * factorCap0, 0.70 * factorCap0);		
			pObj->translate(0,0,-3-offset);

			printf("Done \n");

			printf("- Transfering Internal Cap Geometry to GPU...");
			copyInternalCap0(pObj, offset);
			printf("Done \n");
		}

		///		1. Internal Cap 1
		{
			ObjObject *pObj = new ObjObject(0, 0, 0, 0);	
			pObj->load("nanorod.obj");
			pObj->setSmooth(false);
			
			vect3d spec(0.3,0.3,0.3);
			vect3d diff(0.3,0.3,0.3);
			vect3d ambi(0.3, 0.3, 0.3);
			pObj->setMaterial(spec, diff, ambi, 70);

			vect3d axis(1, 0, 0);
			float angle = -90;
			pObj->rotate(angle, axis);
			pObj->scale(0.82 * factorCap1, 0.82 * factorCap1, 0.70 * factorCap1);		
			pObj->translate(0,0,-3-offset);

			printf("Done \n");

			printf("- Transfering Internal Cap Geometry to GPU...");
			copyInternalCap1(pObj, offset);
			printf("Done \n");
		}

		///		Load nanoPlane
		{
			ObjObject *pObj = new ObjObject(0, 0, 0, 0);	
#ifndef DATA_2D
			pObj->load("nanoPlane.obj");
#else
			pObj->load("2d_plane.obj");
#endif
			pObj->setSmooth(false);
			
			vect3d spec(0.3,0.3,0.3);
			vect3d diff(0.3,0.3,0.3);
			vect3d ambi(1,1,1);
			pObj->setMaterial(spec, diff, ambi, 70);
#ifndef DATA_2D
			vect3d axis(1, 0, 0);
			float angle = -90;
			pObj->rotate(angle, axis);
			pObj->scale(0.82 * factorPlane * 1.01, 0.82 * factorPlane * 1.01, 0.70 * factorPlane * 1.01);		
			pObj->translate(0,0,-3 - offset);
#else
			pObj->scale(0.96,1,1);		
			pObj->translate(-4.5,0, 3);
#endif
			printf("Done \n");

			printf("- Transfering NanoPlane Geometry to GPU...");
			copyNanoPlane(pObj, offset);
			printf("Done \n");
		}
	}
	
	color_map_choice_callback(0);

#ifndef DATA_2D
	//%%%Haptics
	initHaptic();
	//%%%

	setupShaders();

#endif
	///		Go
	nTick = clock();
	glutMainLoop();


	//%%%
	closeHaptic();
	//%%%

	destroy();

	return EXIT_SUCCESS;
}


