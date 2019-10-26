ROOT Hips
JOINT LHipJoint
	JOINT LeftUpLeg
		JOINT LeftLeg
			JOINT LeftFoot
				JOINT LeftToeBase
JOINT RHipJoint
	JOINT RightUpLeg
		JOINT RightLeg
			JOINT RightFoot
				JOINT RightToeBase
JOINT LowerBack
	JOINT Spine
		JOINT Spine1
			JOINT Neck
				JOINT Neck1
					JOINT Head
			JOINT LeftShoulder
				JOINT LeftArm
						JOINT LeftForeArm
							JOINT LeftHand
								JOINT LeftFingerBase
									JOINT LeftHandIndex1
								JOINT LThumb
				JOINT RightShoulder
					JOINT RightArm
						JOINT RightForeArm
							JOINT RightHand
								JOINT RightFingerBase
									JOINT RightHandIndex1
								JOINT RThumb
"""

"""
JOINT_NUM = 31

SDR_L, SDR_R, HIP_L, HIP_R = 18, 25, 2, 7
FOOT_L = [4,5]
FOOT_R = [9,10]
HEAD = 16
FILTER_OUT = []
JOINT_SCALE = 5.644

JOINT_WEIGHTS = [
    1,
    1e-10, 1, 1, 1, 1,
    1e-10, 1, 1, 1, 1,
    1e-10, 1, 1,
    1e-10, 1, 1,
    1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10,
    1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10 ]
"""

"""
ROOT
JOINT Hip_R
  JOINT HipPart1_R
    JOINT HipPart2_R
      JOINT Knee_R
        JOINT KneePart1_R
          JOINT KneePart2_R
            JOINT Ankle_R
              JOINT Toes_R
                JOINT ToesEnd_R
JOINT Spine1_M
  JOINT Spine2_M
    JOINT Spine3_M
      JOINT Chest_M
        JOINT Scapula_R
          JOINT Shoulder_R
            JOINT ShoulderPart1_R
              JOINT ShoulderPart2_R
                JOINT Elbow_R
                  JOINT ElbowPart1_R
                    JOINT ElbowPart2_R
                      JOINT Wrist_R
         JOINT Neck_M
           JOINT NeckPart1_M
             JOINT Head_M
               JOINT HeadEnd_M
               JOINT Head_M_spare
          JOINT Scapula_L
            JOINT Shoulder_L
              JOINT ShoulderPart1_L
                JOINT ShoulderPart2_L
                  JOINT Elbow_L
                    JOINT ElbowPart1_L
                      JOINT ElbowPart2_L
                        JOINT Wrist_L
  JOINT Hip_L
    JOINT HipPart1_L
      JOINT HipPart2_L
        JOINT Knee_L
          JOINT KneePart1_L
            JOINT KneePart2_L
              JOINT Ankle_L
                JOINT Toes_L
                  JOINT ToesEnd_L


"""
JOINT_NUM = 44

SDR_L, SDR_R, HIP_L, HIP_R = 28, 15, 35, 1
FOOT_L = [41,43]
FOOT_R = [7,9]
HEAD = 24
FILTER_OUT = ["Cup", "Finger", "Head_rivet", "Teeth", "Tongue", "Eye", "Pupil", "Iris", "muscleDriver"]
JOINT_SCALE = 1

JOINT_WEIGHTS = [
    1,
    1, 1e-10, 1e-10, 1, 1e-10, 1e-10, 1, 1e-10, 1,
	1, 1, 1, 1,
	            1e-10, 1, 1e-10, 1e-10, 1, 1e-10, 1e-10, 1,
				1, 1e-10, 1, 1e-10, 1e-10,
	            1e-10, 1, 1e-10, 1e-10, 1, 1e-10, 1e-10, 1,
    1, 1e-10, 1e-10, 1, 1e-10, 1e-10, 1, 1e-10, 1 ]
"""
