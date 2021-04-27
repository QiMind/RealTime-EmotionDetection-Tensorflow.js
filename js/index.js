let mobilenet;
let model;

const webcame = new webcam(document.getElementById('webcam'));
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

async function train(){
    dataset.ys = null;
    dataset.encodeLabels(5);

    model = tf.sequential({
        layers: [
            // YOUR CODE HERE
            tf.layers.flatten({ inputShape: mobilenet.outputs[0].shape.slice(1) }),
            tf.layers.dense({ units: 100, activation: 'relu' }),
            tf.layers.dense({ units: 5, activation: 'softmax' })

        ]
    });

    const optimizer = tf.train.adam(0.0001);

    model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

    let loss = 0;
    model.fit(dataset.xs, dataset.ys,{
        epoch
    })
}
