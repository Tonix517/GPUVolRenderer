#include "cCameraVol.h"

void cCameraVol::renderView(const int a_windowWidth, const int a_windowHeight, const int a_imageIndex)
{
   // store most recent size of display
    m_lastDisplayWidth = a_windowWidth;
    m_lastDisplayHeight = a_windowHeight;

    // set background color
    cColorf color = getParentWorld()->getBackgroundColor();
    glClearColor(color.getR(), color.getG(), color.getB(), color.getA());

    // clear the color and depth buffers
    glClear(GL_DEPTH_BUFFER_BIT);

    // compute global pose
    computeGlobalCurrentObjectOnly(true);

    // check window size
    if (a_windowHeight == 0) { return; }

    // render the 'back' 2d object layer; it will set up its own
    // projection matrix
    if (m_back_2Dscene.getNumChildren())
      render2dSceneGraph(&m_back_2Dscene,a_windowWidth,a_windowHeight);    

    // set up perspective projection
    double glAspect = ((double)a_windowWidth / (double)a_windowHeight);

    // set the perspective up for monoscopic rendering
    // Set up the projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    gluPerspective(
            m_fieldViewAngle,   // Field of View Angle.
            glAspect,           // Aspect ratio of viewing volume.
            m_distanceNear,     // Distance to Near clipping plane.
            m_distanceFar);     // Distance to Far clipping plane.


    // Now set up the view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // render pose
    cVector3d lookAt = m_globalRot.getCol0();
    cVector3d lookAtPos;
    m_globalPos.subr(lookAt, lookAtPos);
    cVector3d up = m_globalRot.getCol2();

    gluLookAt( m_globalPos.x,    m_globalPos.y,   m_globalPos.z,
               lookAtPos.x,    lookAtPos.y,   lookAtPos.z,
               up.x,           up.y,          up.z );

	// Back up the projection matrix for future reference
	glGetDoublev(GL_PROJECTION_MATRIX,m_projectionMatrix);

    // Set up reasonable default OpenGL state
	glEnable(GL_LIGHTING);
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    // optionally perform multiple rendering passes for transparency
    if (m_useMultipassTransparency) {
      m_parentWorld->renderSceneGraph(CHAI_RENDER_MODE_NON_TRANSPARENT_ONLY);
      m_parentWorld->renderSceneGraph(CHAI_RENDER_MODE_TRANSPARENT_BACK_ONLY);
      m_parentWorld->renderSceneGraph(CHAI_RENDER_MODE_TRANSPARENT_FRONT_ONLY);
    }
    else
    {
      m_parentWorld->renderSceneGraph(CHAI_RENDER_MODE_RENDER_ALL);
    }        

    // render the 'front' 2d object layer; it will set up its own
    // projection matrix
    if (m_front_2Dscene.getNumChildren())
      render2dSceneGraph(&m_front_2Dscene,a_windowWidth,a_windowHeight);

}