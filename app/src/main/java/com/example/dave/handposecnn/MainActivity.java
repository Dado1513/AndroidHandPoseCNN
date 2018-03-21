package com.example.dave.handposecnn;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.TextureView;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.Arrays;


public class MainActivity extends AppCompatActivity {

    TextureView textureView;
    private static String TAG = MainActivity.class.getName();
    // toDo o change se si sceglie la rete
    public static int width = 150;
    public static int height = 150;
    public static int channel = 1;
    private Button startClassification;
    CameraHandler cameraHandler;
    private ArrayList<String> label ;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);
        startClassification = (Button)findViewById(R.id.btn_handpose);
        textureView = (TextureView)findViewById(R.id.texture);

        label = new ArrayList<>();

        createLabel(5);

        cameraHandler = new CameraHandler(textureView,MainActivity.this, width,height,channel,label);

        startClassification.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cameraHandler.setInit(true);
                Log.e(TAG,"Init Classification");
            }
        });
    }


    @Override
    protected void onPause() {
        Log.e(TAG, "onPause");
        this.cameraHandler.closeCamera();
        this.cameraHandler.stopBackgroundThread();
        this.cameraHandler.setInit(false);
        super.onPause();
    }

    protected void onResume() {
        super.onResume();
        Log.e(TAG, "onResume");
        this.cameraHandler.startBackgroundThread();
        if (textureView.isAvailable()) {
            this.cameraHandler.openCamera();
        } else {
            this.cameraHandler.setListener();
        }
    }

    private void createLabel(int numberLabel){
        for(int i = 1; i < numberLabel +1; i++){
            label.add(String.valueOf(i));
        }
    }

}
