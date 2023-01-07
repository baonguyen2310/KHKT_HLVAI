const detectorConfig = {
  modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER,
};
const detector = await poseDetection.createDetector(
  poseDetection.SupportedModels.MoveNet,
  detectorConfig
);

const poseClassifier = await tf.loadLayersModel(
  "https://models.s3.jp-tok.cloud-object-storage.appdomain.cloud/model.json"
);

const video = document.getElementById("video");
video.style.transform = "rotateY(180deg)";
const outputCanvas = document.getElementById("outputCanvas");
outputCanvas.style.transform = "rotateY(180deg)";

const draw = (canvas, video, poses) => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0);
  for (let i = 0; i < 17; i++) {
    const x = poses[0].keypoints[i].x;
    const y = poses[0].keypoints[i].y;
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    ctx.fillStyle = "gray";
    ctx.fill();
    ctx.stroke();
  }
  const couplePoints = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [5, 6],
    [6, 8],
    [8, 10],
    [5, 7],
    [7, 9],
    [6, 12],
    [12, 14],
    [14, 16],
    [5, 11],
    [11, 13],
    [13, 15],
    [11, 12],
  ];
  for (let i = 0; i < couplePoints.length; i++) {
    const x1 = poses[0].keypoints[couplePoints[i][0]].x;
    const y1 = poses[0].keypoints[couplePoints[i][0]].y;
    const x2 = poses[0].keypoints[couplePoints[i][1]].x;
    const y2 = poses[0].keypoints[couplePoints[i][1]].y;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }
};

const CLASS_NO = {
  Chair: 0,
  Cobra: 1,
  Dog: 2,
  No_Pose: 3,
  Shoulderstand: 4,
  Traingle: 5,
  Tree: 6,
  Warrior: 7,
};

const NO_CLASS = ["Chair", "Cobra", "Dog", "No_Pose", "Shoulderstand", "Traingle", "Tree", "Warrior"];

const POINTS = {
    NOSE : 0,
    LEFT_EYE : 1,
    RIGHT_EYE : 2,
    LEFT_EAR : 3,
    RIGHT_EAR : 4,
    LEFT_SHOULDER : 5,
    RIGHT_SHOULDER : 6,
    LEFT_ELBOW : 7,
    RIGHT_ELBOW : 8,
    LEFT_WRIST : 9,
    RIGHT_WRIST : 10,
    LEFT_HIP : 11,
    RIGHT_HIP : 12,
    LEFT_KNEE : 13,
    RIGHT_KNEE : 14,
    LEFT_ANKLE : 15,
    RIGHT_ANKLE : 16,
}

function get_center_point(landmarks, left_bodypart, right_bodypart) {
  let left = tf.gather(landmarks, left_bodypart, 1);
  let right = tf.gather(landmarks, right_bodypart, 1);
  const center = tf.add(tf.mul(left, 0.5), tf.mul(right, 0.5));
  return center;
}

function get_pose_size(landmarks, torso_size_multiplier = 2.5) {
  let hips_center = get_center_point(
    landmarks,
    POINTS.LEFT_HIP,
    POINTS.RIGHT_HIP
  );
  let shoulders_center = get_center_point(
    landmarks,
    POINTS.LEFT_SHOULDER,
    POINTS.RIGHT_SHOULDER
  );
  let torso_size = tf.norm(tf.sub(shoulders_center, hips_center));
  let pose_center_new = get_center_point(
    landmarks,
    POINTS.LEFT_HIP,
    POINTS.RIGHT_HIP
  );
  pose_center_new = tf.expandDims(pose_center_new, 1);

  pose_center_new = tf.broadcastTo(pose_center_new, [1, 17, 2]);
  // return: shape(17,2)
  let d = tf.gather(tf.sub(landmarks, pose_center_new), 0, 0);
  let max_dist = tf.max(tf.norm(d, "euclidean", 0));

  // normalize scale
  let pose_size = tf.maximum(
    tf.mul(torso_size, torso_size_multiplier),
    max_dist
  );
  return pose_size;
}

function normalize_pose_landmarks(landmarks) {
  let pose_center = get_center_point(
    landmarks,
    POINTS.LEFT_HIP,
    POINTS.RIGHT_HIP
  );
  pose_center = tf.expandDims(pose_center, 1);
  pose_center = tf.broadcastTo(pose_center, [1, 17, 2]);
  landmarks = tf.sub(landmarks, pose_center);

  let pose_size = get_pose_size(landmarks);
  landmarks = tf.div(landmarks, pose_size);
  return landmarks;
}

function landmarks_to_embedding(landmarks) {
  // normalize landmarks 2D
  landmarks = normalize_pose_landmarks(tf.expandDims(landmarks, 0));
  let embedding = tf.reshape(landmarks, [1, 34]);
  return embedding;
}

function predictWebcam() {
  detector.estimatePoses(video).then((poses) => {
    draw(outputCanvas, video, poses);
    const keypoints = poses[0].keypoints;
    let input = keypoints.map((keypoint) => {
        return [keypoint.x, keypoint.y]
    })
    const processedInput = landmarks_to_embedding(input)
    const classification = poseClassifier.predict(processedInput)
    classification.array().then((data) => {
        for(let i = 0; i < 8; i++){
            if (data[0][i] > 0.97){
                console.log(NO_CLASS[i]);
            }
        }
    })
  });
  window.requestAnimationFrame(predictWebcam);
}

if (!!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
  const constraints = {
    video: {
      facingMode: "user",
    },
  };
  navigator.mediaDevices.getUserMedia(constraints).then((mediastream) => {
    video.srcObject = mediastream;
    video.addEventListener("loadeddata", predictWebcam);
    video.onloadedmetadata = () => {
      video.play();
    };
  });
} else {
  console.warn("getUserMedia() is not supported by your browser");
}
