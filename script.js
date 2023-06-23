let json_data;
let model;

// Upload model
$('#upload-model').click(function() {
    $('<input type="file">').on('change', function(event) {
    let file = event.target.files[0];
    let reader = new FileReader();
    
    reader.onload = function(e) {
        let modelData = e.target.result;
        tf.loadLayersModel(modelData).then(function(model) {
            logData('Модель успішно завантажена: <br>' + model);
        }).catch(function(error) {
            console.log(error);
            logData('Сталася помилка при завантаженні моделі.');
        });
    };
    
    reader.readAsArrayBuffer(file);
    }).click();
});

// Download model
$('#download-model').click(function() {
    try {
        model.save('downloads://deepthinker_model').then(function() {
        logData('Модель успішно збережена');
        }).catch(function(error) {
        logData('Помилка при збереженні моделі: <br>' + error);
        });
    } catch (error) {
        console.log(error);
        logData('Не вдалося завантажити модель. Можливо, ви її ще не створили.')
    }
});

// Update file data
$("#dataset-file").change(async function(e) {
    e.preventDefault();
    $("#output-section").html("");
    $("#input-values").html(`
        <legend><div class="tooltip">?<span class="tooltiptext"><i>Input values</i> (вхідні значення) представляють собою дані або фічі, які подаються на вхід моделі для обробки. Вони є вхідними точками, через які дані проходять усередину мережі для обчислення та передачі інформації.</span></div>
        Input values:</legend>`);
    $("#output-values").html(`
        <legend><div class="tooltip">?<span class="tooltiptext"><i>Output values</i> (вихідні значення) - це результати, отримані на виході мережі після обробки вхідних даних. Коли дані проходять через нейронну мережу, вони проходять крізь різні шари нейронів, де виконуються обчислення. </span></div>
        Output values:</legend>`);

    if (e.target.files[0].type !== "application/json") {
        logData("Будь ласка, завантажте файл у форматі .json");
        return;
    }

    const file_data = await getFileData(e.target.files[0]);
    json_data = JSON.parse(file_data);
    valuesCheckboxes(Object.keys(json_data[0]));
})

// Get file data
async function getFileData(file) {
    let reader = new FileReader();
    reader.readAsText(file, "UTF-8");
    await new Promise(resolve => reader.onload = () => resolve());
    return reader.result;
}

// Create values checkboxes
function valuesCheckboxes(features) {
    for (let i of features) {
        $("#input-values").append(
            `<input type="checkbox" id="${i}" name="input-value" value="${i}">
            <label for="${i}">${i}</label><br>`
        );

        $("#output-values").append(
            `<input type="checkbox" id="${i}" name="output-value" value="${i}">
            <label for="${i}">${i}</label><br>`
        );
    }
}

// Add blank layer
$("#add-layer").click(function (e) {
    e.preventDefault();
    const num = $(".layer").length + 1;

    const layerHtml = `
        <div class="arrow" id="arr${num}">⬇<br></div>
        <div class="layer" id="layer${num}">
            Layer ${num} | <input class="number-parameter" type="number" id="units${num}" name="units${num}" min="1" max="100" required> units<br>
            <div class="row activation">
                <label for="activation${num}">Activation >></label><br>
                <select class="number-parameter" name="activation${num}" id="activation${num}">
                    <option value="">None (Linear Activation)</option>
                    <option value="relu">Rectified Linear Unit (ReLU)</option>
                    <option value="sigmoid">Sigmoid</option>
                    <option value="tanh">Hyperbolic Tangent (Tanh)</option>
                    <option value="softplus">Softplus</option>
                </select>
            </div>
        </div>`;

    document.getElementById("layers").insertAdjacentHTML("beforeend", layerHtml);
    $("#remove-layer").css("visibility", "visible");
    
    if (num >= 10) {
        $("#add-layer").css("visibility", "hidden");
    }
})

// Remove last layer
$("#remove-layer").click(function(e) {
    e.preventDefault();
    const num = $(".layer").length;

    $(`#layer${num}`).remove();
    $(`#arr${num}`).remove();

    if (num <= 2) $("#remove-layer").css("visibility", "hidden");
    if (num <= 10) $("#add-layer").css("visibility", "visible");;
})

// Use default params
$("#default").click(function(e) {
    e.preventDefault();
    $("#learning-rate")[0].value = $("#learning-rate")[0].placeholder;
    $("#validation-split")[0].value = $("#validation-split")[0].placeholder;
    $("#batch-size")[0].value = $("#batch-size")[0].placeholder;
    $("#epochs")[0].value = $("#epochs")[0].placeholder;
    $("#units1")[0].value = $("#units1")[0].placeholder;
})

// Run model
$("#model-params").submit(function(e) {
    e.preventDefault();
    const model_data = getFormData();
    console.log(model_data);

    tf.setBackend('webgl');
    console.log("BACKEND: ", tf.getBackend());

    tf.util.shuffleCombo(model_data.input_data, model_data.output_data);
    const INPUTS_TENSOR = tf.tensor2d(model_data.input_data);
    const OUTPUTS_TENSOR = tf.tensor2d(model_data.output_data);
    logData("Normalizing values...");
    const FEATURE_RESULTS = normalize(INPUTS_TENSOR);
    logData("Min Values:<br>" + FEATURE_RESULTS.MIN_VALUES);
    logData("Max Values:<br>" + FEATURE_RESULTS.MAX_VALUES);

    //INPUTS_TENSOR.dispose();

    logData("Creating a model...");
    model = createModel(
        model_data.input_values.length,
        model_data.output_values.length, 
        model_data.layers);
    
    model.summary()
    // tfvis.show.modelSummary({name: 'Model Summary'}, model);

    logData("Model training...");
    train(
        model,
        model_data["learning-rate"], 
        model_data["loss"], 
        FEATURE_RESULTS, 
        OUTPUTS_TENSOR, 
        model_data["validation-split"], 
        model_data["batch-size"], 
        model_data["epochs"]);

    addPredictFields(model_data.input_values);

    //OUTPUTS_TENSOR.dispose();
    //FEATURE_RESULTS.NORMALIZED_VALUES.dispose();

    // evaluate();

    //FEATURE_RESULTS.MIN_VALUES.dispose();
    //FEATURE_RESULTS.MAX_VALUES.dispose();
    //model.dispose();
})

// Get data from user
function getFormData() {
    $("#output-section").html("");
    const input_values = [];
    const output_values = [];

    $('input[name="input-value"]:checked').each(function() {
        input_values.push(this.value);
    });

    $('input[name="output-value"]:checked').each(function() {
        output_values.push(this.value);
    });

    if (input_values.length == 0) {
        logData("Оберіть хоча б одне вхідне значення");
        return;
    }

    if (output_values.length == 0) {
        logData("Оберіть хоча б одне вихідне значення");
        return;
    }

    let input_data = json_data.map(obj =>
        input_values.map(prop => obj[prop]));

    let output_data = json_data.map(obj =>
        output_values.map(prop => obj[prop]));

    const data = {
        "input_data": input_data,
        "input_values": input_values,
        "output_data": output_data,
        "output_values": output_values,
        "learning-rate": Number($("#learning-rate")[0].value),
        "validation-split": Number($("#validation-split")[0].value),
        "batch-size": Number($("#batch-size")[0].value),
        "epochs": Number($("#epochs")[0].value),
        "loss": $("#loss")[0].value,
        "layers": $(".layer").toArray().map((el) => ({
            "units": Number(el.childNodes[1].value),
            "activation": el.childNodes[5].childNodes[4].value
        }))
    }

    return data;
}

// Normalize data
function normalize(tensor, min, max) {

    const result = tf.tidy(function () {
        const MIN_VALUES = min || tf.min(tensor, 0);
        const MAX_VALUES = max || tf.max(tensor, 0);
        const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);
        const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);
        const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);
        return { NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES };
    });

    return result;
}

// Create model
function createModel(input_value_length, output_value_length, layers) {
    const model = tf.sequential();

    console.log(layers)
    let i = layers[0]
    if (i.activation == '') {
        model.add(tf.layers.dense({inputShape: [input_value_length], units: Number(i.units)}));
    } else {
        model.add(tf.layers.dense({inputShape: [input_value_length], units: Number(i.units), activation: i.activation}));
        }

    for (let i of layers.slice(1)) {
        if (i.activation == '') {
            model.add(tf.layers.dense({units: Number(i.units)}));
        } else {
            model.add(tf.layers.dense({units: Number(i.units), activation: i.activation}));
        }
    }

    model.add(tf.layers.dense({units: output_value_length}));
    return model;
}

// Train model
async function train(model, LEARNING_RATE, LOSS, FEATURE_RESULTS, OUTPUTS_TENSOR, validation_split, batch_size, epochs) {
    console.log("FUNCTION PARAMS", model, LEARNING_RATE, LOSS, FEATURE_RESULTS, OUTPUTS_TENSOR, validation_split, batch_size, epochs)
    model.compile({
        optimizer: tf.train.sgd(LEARNING_RATE),
        loss: LOSS,
    });

    let results = await model.fit(
        FEATURE_RESULTS.NORMALIZED_VALUES,
        OUTPUTS_TENSOR,
        {
            validationSplit: validation_split,
            shuffle: true,
            batchSize: batch_size,
            epochs: epochs
        });

    logData(
        "Average error loss:<br>" +
        Math.sqrt(results.history.loss[results.history.loss.length - 1])
    );
    logData(
        "Average validation error loss:<br>" +
        Math.sqrt(results.history.val_loss[results.history.val_loss.length - 1])
    );
}

// Callback function
function onBatchEnd(batch, logs) {
    logData('Accuracy: ' + logs.acc);
  }

//   // Predict values
// function evaluate() {
//     tf.tidy(function () {
//         let newInput = normalize(
//             tf.tensor2d([[750, 1]]),
//             FEATURE_RESULTS.MIN_VALUES,
//             FEATURE_RESULTS.MAX_VALUES
//         );

//         let output = model.predict(newInput.NORMALIZED_VALUES);
//         output.print();
//     });

//     logData(tf.memory().numTensors);
// }

// Add new fields for prediction
function addPredictFields(input_values) {
    $("#predict-fields").html("");
    $("#predict-button").html("");
    for (let i of input_values) {
        $("#predict-fields").append(
            `<div class="predict-value"><label for="predict_${i}">${i}</label>
            <input type="text" id="predict_${i}" name="predict_${i}" class="number-parameter predict-input"><br></div>`
        );
    }

    $("#predict-button").append(
        `<div class="button smal centrify">Predict</div>`
    );
}

// Predict values
$('#predict-button').click(function(e) {
    e.preventDefault();
    try {
        let predict_inputs = $(".predict-input")
        let predict_values = []
        for (let i of predict_inputs) {
            predict_values.push(Number(i.value))
        }
        logData("Input values:<br>" + JSON.stringify(predict_values));
        let output = model.predict(tf.tensor2d([predict_values]));
        console.log(output);
        logData("Predicted values:<br>" + JSON.stringify(Array.from(output.dataSync())))
    } catch (error) {
        console.log(error);
        logData('Вхідні значення для передбачення нових значень введені некоректно')
    }
});

// Log data
function logData(data) {
    $("#output-section").append(`<br><span>${data}</span><br>`);
}

// Open hints
$("#open-hints").click(function(e) {
    e.preventDefault();
    $("#hints-list").slideToggle("slow");
});
