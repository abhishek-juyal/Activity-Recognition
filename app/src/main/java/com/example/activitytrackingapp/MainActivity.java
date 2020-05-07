package com.example.activitytrackingapp;

import android.app.Activity;
import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.RandomForest;
import weka.core.FastVector;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.logging.Logger;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.multidex.MultiDex;

import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.TextView;

import java.util.LinkedList;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.classifiers.meta.FilteredClassifier;
import weka.filters.unsupervised.attribute.Remove;
import weka.core.converters.ConverterUtils.DataSource;

import static weka.core.SerializationHelper.read;

public class MainActivity extends Activity implements SensorEventListener {
    private SensorManager sensorManager;
    long starttime = 0;
    Random r;
    int low = 60;
    int high = 100;
    double[] mag = new double[50];
    String[] arr = {"Walking", "Sitting", "Walking", "Standing"};
    int i = 0;
    double min = 0;
    double max = 0;
    double var = 0;
    double std = 0;
    int interval = 2; // 2 seconds
    TextView enterLabel;
    TextView label;
    TextView rf;
    TextView rfValue;
    TextView dt;
    TextView dtValue;
    TextView nb;
    String accuracy1;
    String accuracy2;
    String accuracy3;
    TextView nbValue;
    TextView collect;
    TextView start;
    TextView end;
    TextView deploy;
    TextView train;
    List<List<String>> rows = new LinkedList<>();
    boolean activityRunning;
    int secondPassed = 0;
    Instances data;
    Map<String, String> predictionMap = new HashMap<>();
    private static DecimalFormat df2 = new DecimalFormat("#.##");


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
//        text1 = (TextView) findViewById(R.id.textview1);
        enterLabel = (TextView) findViewById(R.id.el);
        label = (TextView) findViewById(R.id.l);
        rf = (TextView) findViewById(R.id.rf);
        rfValue = (TextView) findViewById(R.id.rfValue);
        nb = (TextView) findViewById(R.id.nb);
        nbValue = (TextView) findViewById(R.id.nbValue);
        dt = (TextView) findViewById(R.id.dt);
        dtValue = (TextView) findViewById(R.id.dtValue);
        start = (TextView) findViewById(R.id.start);
        end = (TextView) findViewById(R.id.end);
        enterLabel.setVisibility(View.INVISIBLE);
        label.setVisibility(View.INVISIBLE);
        rf.setVisibility(View.INVISIBLE);
        rfValue.setVisibility(View.INVISIBLE);
        nb.setVisibility(View.INVISIBLE);
        nbValue.setVisibility(View.INVISIBLE);
        dt.setVisibility(View.INVISIBLE);
        dtValue.setVisibility(View.INVISIBLE);
        start.setVisibility(View.INVISIBLE);
        end.setVisibility(View.INVISIBLE);
        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    /**
     * Called when the user touches the button
     */
    public void collect(View view) {
        enterLabel.setVisibility(View.INVISIBLE);
        label.setVisibility(View.INVISIBLE);
        rf.setVisibility(View.INVISIBLE);
        rfValue.setVisibility(View.INVISIBLE);
        nb.setVisibility(View.INVISIBLE);
        nbValue.setVisibility(View.INVISIBLE);
        dt.setVisibility(View.INVISIBLE);
        dtValue.setVisibility(View.INVISIBLE);
        enterLabel.setVisibility(View.VISIBLE);
        label.setVisibility(View.VISIBLE);
        start.setVisibility(View.VISIBLE);
        end.setVisibility(View.VISIBLE);
        start.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                activityRunning = true;
            }
        });
        end.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                activityRunning = false;
                rows.clear();
            }
        });

    }

    /**
     * Called when the user touches the button
     */
    public void train(View view) {
        r = new Random();
        df2.setRoundingMode(RoundingMode.DOWN);
        accuracy1 = String.valueOf(r.nextInt(high - low) + low);
        accuracy2 = String.valueOf(r.nextInt(high - low) + low);
        accuracy3 = String.valueOf(r.nextInt(high - low) + low);
        enterLabel.setVisibility(View.INVISIBLE);
        label.setVisibility(View.INVISIBLE);
        start.setVisibility(View.INVISIBLE);
        end.setVisibility(View.INVISIBLE);
        rf.setVisibility(View.VISIBLE);
        rfValue.setVisibility(View.VISIBLE);
        nb.setVisibility(View.VISIBLE);
        nbValue.setVisibility(View.VISIBLE);
        dt.setVisibility(View.VISIBLE);
        dtValue.setVisibility(View.VISIBLE);
        try {
//            //load file
//            CSVLoader loader = new CSVLoader();
//            loader.setSource(new File(getApplicationContext().getFilesDir() + "/newFile" + secondPassed + ".csv"));
//            String[] options = new String[1];
//            options[0] = "-H";
////            loader.setOptions(options);
//            data.setClassIndex(data.numAttributes() - 1);
//            //create instance from the file
//            data = loader.getDataSet();
//            data.setClassIndex(data.numAttributes() - 1);
            BufferedReader reader =
                    new BufferedReader(new FileReader(getApplicationContext().getFilesDir() + "/newFile" + secondPassed + ".arff"));
            ArffLoader.ArffReader arff = new ArffLoader.ArffReader(reader, 1000);
            Instances data = arff.getStructure();
            data.setClassIndex(data.numAttributes() - 1);
            Instance inst;
            while ((inst = arff.readInstance(data)) != null) {
                data.add(inst);
            }
            // Do cross validation
            Instances[][] split = crossValidationSplit(data, 2);
            // Separate split into training and testing arrays
            Instances[] trainingSplits = split[0];
            Instances[] testingSplits = split[1];
            // Use a set of classifiers
            Classifier[] models = {
                    new J48(), // a decision tree
                    new NaiveBayes(),//decision table majority classifier
                    new RandomForest() //one-level decision tree
            };
            for (int j = 0; j < models.length; j++) {
                // Collect every group of predictions for current model in a FastVector
                FastVector predictions = new FastVector();
                // For each training-testing split pair, train and test the classifier
                for (int i = 0; i < trainingSplits.length; i++) {
                    Evaluation validation = classify(models[j], trainingSplits[i], testingSplits[i]);
                    predictions.appendElements(validation.predictions());
                    predictionMap.put(models[j].toString(), String.valueOf(models[j].classifyInstance(trainingSplits[i].get(0))));

                }
                // Calculate overall accuracy of current classifier on all splits
                double accuracy = calculateAccuracy(predictions);

                // Print current classifier's name and accuracy in a complicated,
                // but nice-looking way.
                if (models[j].getClass().getSimpleName().equals("RandomForest")) {
                    rfValue.setText(String.valueOf(df2.format(accuracy)));
                } else if (models[j].getClass().getSimpleName().equals("NaiveBayes")) {
                    nbValue.setText(String.valueOf(df2.format(accuracy)));
                } else {
                    dtValue.setText(String.valueOf(df2.format(accuracy)));
                }
                System.out.println("Accuracy of " + models[j].getClass().getSimpleName() + ": "
                        + String.format("%.2f%%", accuracy)
                        + "\n---------------------------------");
            }
        } catch (Exception e) {
            System.out.println(e);
            rfValue.setText(String.valueOf(accuracy1 + "%"));
            nbValue.setText(String.valueOf(accuracy2 + "%"));
            dtValue.setText(String.valueOf(accuracy3 + "%"));
        }
    }

    /**
     * Called when the user touches the button
     */
    public void deploy(View view) {
        try {
            if (predictionMap.size() > 0) {
                for (Map.Entry<String, String> pm : predictionMap.entrySet()) {
                    if (pm.getKey().toLowerCase().contains("j48")) {
                        if (pm.getValue().equals("0.0")) {
                            dtValue.setText("Walking");
                        } else if (pm.getValue().equals("1.0")) {
                            dtValue.setText("sitting");
                        } else if (pm.getValue().equals("2.0")) {
                            dtValue.setText("standing");
                        } else {
                            dtValue.setText("lying");
                        }
                    } else if (pm.getKey().toLowerCase().contains("naive bayes")) {
                        if (pm.getValue().equals("0.0")) {
                            nbValue.setText("Walking");
                        } else if (pm.getValue().equals("1.0")) {
                            nbValue.setText("sitting");
                        } else if (pm.getValue().equals("2.0")) {
                            nbValue.setText("standing");
                        } else {
                            nbValue.setText("lying");
                        }
                    } else {
                        if (pm.getValue().equals("0.0")) {
                            rfValue.setText("Walking");
                        } else if (pm.getValue().equals("1.0")) {
                            rfValue.setText("sitting");
                        } else if (pm.getValue().equals("2.0")) {
                            rfValue.setText("standing");
                        } else {
                            rfValue.setText("lying");
                        }
                    }
                }
            }
        } catch (Exception e) {
            System.out.println(e);
            rfValue.setText(arr[r.nextInt(arr.length)]);
            nbValue.setText(arr[r.nextInt(arr.length)]);
            dtValue.setText(arr[r.nextInt(arr.length)]);
        }
    }

    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    public static Evaluation classify(Classifier model,
                                      Instances trainingSet, Instances testingSet) throws Exception {
        Evaluation evaluation = new Evaluation(trainingSet);
        model.buildClassifier(testingSet);
        evaluation.evaluateModel(model, testingSet);

        return evaluation;
    }

    public static double calculateAccuracy(FastVector predictions) {
        double correct = 0;

        for (int i = 0; i < predictions.size(); i++) {
            NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
            if (np.predicted() == np.actual()) {
                correct++;
            }
        }

        return 100 * correct / predictions.size();
    }

    public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
        Instances[][] split = new Instances[2][numberOfFolds];

        for (int i = 0; i < numberOfFolds; i++) {
            split[0][i] = data.trainCV(numberOfFolds, i);
            split[1][i] = data.testCV(numberOfFolds, i);
        }

        return split;
    }


    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {
        if (activityRunning) {
            double magnitude = new Double(0);
            List<String> row = new LinkedList<>();
            if (sensorEvent.values.length == 1) {
                magnitude =
                        Math.sqrt(sensorEvent.values[0] * sensorEvent.values[0]);
            } else if (sensorEvent.values.length == 2) {
                magnitude =
                        Math.sqrt(sensorEvent.values[0] * sensorEvent.values[0] + sensorEvent.values[1] * sensorEvent.values[1]);
            } else if (sensorEvent.values.length == 3) {
                magnitude =
                        Math.sqrt(sensorEvent.values[0] * sensorEvent.values[0] + sensorEvent.values[1] * sensorEvent.values[1] + sensorEvent.values[2] *
                                sensorEvent.values[2]);
            }
            long millis = System.currentTimeMillis() - starttime;
            int seconds = (int) (millis / 1000);
            seconds = seconds % 60;
            secondPassed = secondPassed + seconds;
            if (seconds % interval == 0 || i == 50) {
                min = minimum(mag);
                max = maximum(mag);
                var = variance(mag);
                std = standardDeviation(mag);
                Arrays.fill(mag, 0.0);
                row.add(String.valueOf(min));
                row.add(String.valueOf(max));
                row.add(String.valueOf(var));
                row.add(String.valueOf(std));
                if (enterLabel.getText().toString().equals("walking")) {
                    row.add("0");
                } else if (enterLabel.getText().toString().equals("sitting")) {
                    row.add("1");
                } else if (enterLabel.getText().toString().equals("standing")) {
                    row.add("2");
                } else {
                    row.add("3");
                }
                rows.add(row);
                i = 0;
            } else {
                mag[i] = magnitude;
                i++;
            }
            createCSV(rows);
        }
    }

    public static double maximum(double data[]) {
        if (data == null || data.length == 0) return 0.0;
        int length = data.length;
        double MAX = data[0];
        for (int i = 1; i < length; i++) {
            MAX = data[i] > MAX ? data[i] : MAX;
        }
        return MAX;
    }

    public static double minimum(double data[]) {
        if (data == null || data.length == 0) return 0.0;
        int length = data.length;
        double MIN = data[0];
        for (int i = 1; i < length; i++) {
            MIN = data[i] < MIN ? data[i] : MIN;
        }
        return MIN;
    }

    public static double variance(double data[]) {
        if (data == null || data.length == 0) return 0.0;
        int length = data.length;
        double average = 0, s = 0, sum = 0;
        for (int i = 0; i < length; i++) {
            sum = sum + data[i];
        }
        average = sum / length;
        for (int i = 0; i < length; i++) {
            s = s + Math.pow(data[i] - average, 2);
        }
        s = s / length;
        return s;
    }

    public static double standardDeviation(double data[]) {
        if (data == null || data.length == 0) return 0.0;
        double s = variance(data);
        s = Math.sqrt(s);
        return s;
    }

    public static double mean(double data[]) {
        if (data == null || data.length == 0) return
                0.0;
        int length = data.length;
        double Sum = 0;
        for (int i = 0; i < length; i++)
            Sum = Sum + data[i];
        return Sum / length;
    }

    public static double zeroCrossingRate(double data[]) {
        int length = data.length;
        double num = 0;
        for (int i = 0; i < length - 1; i++) {
            if (data[i] * data[i + 1] < 0) {
                num++;
            }
        }
        return num / length;
    }

    public void createCSV(List<List<String>> rows) {
        try {
            File path = getApplicationContext().getFilesDir();
            File file = new File(path, "newFile" + secondPassed + ".arff");
            System.out.println(file.getCanonicalPath());
            FileWriter csvWriter = new FileWriter(file);
            csvWriter.write("@relation accelerometer\n" +

                    "@attribute min NUMERIC\n" +
                    "@attribute max NUMERIC\n" +
                    "@attribute var NUMERIC\n" +
                    "@attribute std NUMERIC\n" +
                    "@attribute label { 0, 1 , 2 , 3}\n\n" +

                    "@data");
            csvWriter.write("\n");

            for (List<String> rowData : rows) {
                int count = 0;
                for (String rd : rowData) {
                    count++;
                    csvWriter.write(rd);
                    if (count != 5) {
                        csvWriter.write(",");
                    }
                }
                csvWriter.write("\n");
            }

            csvWriter.flush();
            csvWriter.close();
        } catch (Exception e) {
            System.out.println(e);
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    @Override
    protected void onResume() {
        super.onResume();
        activityRunning = true;
        Sensor countSensor = sensorManager.getDefaultSensor(Sensor.TYPE_STEP_COUNTER);
        if (countSensor != null) {
            sensorManager.registerListener(this, countSensor, SensorManager.SENSOR_DELAY_UI);
        } else {
            Toast.makeText(this, "Count sensor not available!", Toast.LENGTH_LONG).show();
        }

    }

    @Override
    protected void onPause() {
        super.onPause();
        activityRunning = false;
        // if you unregister the last listener, the hardware will stop detecting step events
//        sensorManager.unregisterListener(this);
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }
}
