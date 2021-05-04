let mobilenet;
let model;

const webcame = new Webcam(document.getElementById('webcam'));
const dataset = new DataSet();

// 1 buna 10 rele
var happySample = 0,
    sadSample = 0,
    //Angry Emotion = scandal On
    scandalOnSample = 0,
    scaredSample = 0,
    calmSample = 0;

let isPredicting = false;

async function loadMobilNet(){
    const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train() {
    dataset.ys = null;
    dataset.encodeLabels(5);

    model = tf.sequential({
        layers: [
            tf.layers.flatten({ inputShape: mobilenet.outputs[0].shape.slice(1) }),
            tf.layers.dense({ units: 100, activation: 'relu' }),
            tf.layers.dense({ units: 5, activation: 'softmax' })

        ]
    });
    const optimizer = tf.train.adam(0.0001);

    model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });
    let loss = 0;
    model.fit(dataset.xs, dataset.ys, {
        epochs: 10,
        callbacks: {
            onBatchEnd: async(batch, logs) => {
                loss = logs.loss.toFixed(5);
                console.log('LOSS: ' + loss);
            }
        }
    });
}


function handleButton(elem){
    switch(elem.id){
        case "0":
            happySample++;
            document.getElementById('happySample').innerText = happySample;
            break;
        case '1':
            sadSample++;
            document.getElementById('sadSample').innerText = sadSample;
            break;
        case "2":
            scandalOnSample++;
            document.getElementById('scandalOnSample').innerText = scandalOnSample;
            break;
        case "3":
            scaredSample++;
            document.getElementById('scaredSample').innerText = scaredSample;
            break;
        case "4":
            calmSample++;
            document.getElementById('calmSample').innerText = calmSample;
            break;
    }

    label = parseInt(elem.id);
    const img = webcame.capture();
    dataset.addExample(mobilenet.predict(img),label);
}

async function predict(){
    while(isPredicting){
        const predictedClass = tf.tidy(() =>{
            const img = webcame.capture();
            const activation = mobilenet.predict(img);
            const predictions = model.predict(activation);
            return predictions.as1D().argMax();
        });

        const classId = ( await predictedClass.data())[0];
        var predictedText = "";

        switch(classId){
            case 0:
                predictedText = "happy";
                break;
            case 1:
                predictedText = 'sad';
                break;
            case 2:
                predictedText = 'angry';
                break;
            case 3:
                predictedText = 'scared';
                break;
            case 4:
                predictedText = 'calm';
                break;
        }

        document.getElementById('prediction').innerText = 'PREDICT : ' +  predictedText;

        predictedClass.dispose();
        await tf.nextFrame();
    }
}

function doTraining(){
    train();
    alert("Training Done");
}

function startPredicting(){
    isPredicting = true;
    document.getElementById('prediction').innerText = 'PREDICT : ';
    predict();
}

function stopPredicting(){
    isPredicting = false;
    predict();
}

async function init(){
    await webcame.setup();
    mobilenet = await loadMobilNet();
    tf.tidy(() => mobilenet.predict(webcame.capture()));
}

init();