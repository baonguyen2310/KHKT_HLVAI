const detectorConfig = {modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING};
const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);

const image = document.getElementById("image_multipose");
const poses = await detector.estimatePoses(image);
console.log(poses);

const outputCanvas = document.getElementById("outputCanvas");

const draw = (canvas, image, poses) => {
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image, 0, 0);
    for (let j = 0; j < poses.length; j++){
        for (let i = 0; i < 17; i++){
            const x = poses[j].keypoints[i].x;
            const y = poses[j].keypoints[i].y;
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = "gray";
            ctx.fill();
            ctx.stroke();
        }
        const couplePoints = [
            [0,1],[0,2],[1,3],[2,4],
            [5,6],[6,8],[8,10],[5,7],[7,9],
            [6,12],[12,14],[14,16],
            [5,11],[11,13],[13,15],[11,12]
        ]
        for(let i = 0; i < couplePoints.length; i++){
            const x1 = poses[j].keypoints[couplePoints[i][0]].x;
            const y1 = poses[j].keypoints[couplePoints[i][0]].y;
            const x2 = poses[j].keypoints[couplePoints[i][1]].x;
            const y2 = poses[j].keypoints[couplePoints[i][1]].y;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        }
    }
}

draw(outputCanvas, image, poses);



