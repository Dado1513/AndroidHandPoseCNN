package com.example.dave.handposecnn;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.AsyncTask;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.util.Size;
import android.util.SparseIntArray;
import android.view.Surface;
import android.view.TextureView;
import android.widget.Toast;

import org.apache.commons.io.FileUtils;
import org.datavec.image.loader.AndroidNativeImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by dave on 16/03/18.
 */

public class CameraHandler {

    private TextureView textureView;
    private long start;
    private static final int REQUEST_CAMERA_PERMISSION = 200;

    private static final String TAG = "HandPoseCNN";
    private Size imageDimension;
    private ImageReader imageReader;
    private String  cameraId;
    protected CameraDevice cameraDevice;
    protected CameraCaptureSession cameraCaptureSessions;
    protected CaptureRequest captureRequest;
    protected CaptureRequest.Builder captureRequestBuilder;
    private AppCompatActivity context;
    private Handler mBackgroundHandler;
    private HandlerThread mBackgroundThread;

    private boolean init = false;
    private String nameFile ;
    private MultiLayerNetwork handposenet;
    private String pathModel ;
    private ArrayList<String> label;
    private boolean netIsLoaded = false;
    private int choose = 1; // choose 1 -->CannyEdge  2--> Canny

    private int idResources = R.raw.handposeclassificationresult92; // id CNN
    private File fileToSaveModel ;

    // toDo o change se si sceglie la rete
    private int width ;
    private int height ;
    private int channel ;
    private Bitmap imageToAnalyze;




    private static final SparseIntArray ORIENTATIONS = new SparseIntArray();

    // questo perchè l'orientazione di camera handler viene presa orizzontale mentre io la voglio verticale
    static {
        ORIENTATIONS.append(Surface.ROTATION_90, 0);
        ORIENTATIONS.append(Surface.ROTATION_0, 90);
        ORIENTATIONS.append(Surface.ROTATION_180, 270);
        ORIENTATIONS.append(Surface.ROTATION_270, 180);
    }



    public CameraHandler(TextureView textureView, Context context, int width, int height, int channel, ArrayList<String> label){
        this.textureView = textureView;
        this.textureView.setSurfaceTextureListener(textureListener);
        this.context = (AppCompatActivity)context;
        this.width = width;
        this.height = height;
        this.channel = channel;
        this.nameFile = "image_to_analyze.jpg";
        this.pathModel ="/model/handposeclassification.zip";
        this.label = label;
        this.fileToSaveModel = new File(Environment.getExternalStorageDirectory()+pathModel);
        new LoadNetworkThread().execute();

    }

    public void setInit(boolean value){
        this.init = value;
    }

    TextureView.SurfaceTextureListener textureListener = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
            //open your camera here
            openCamera();
            start = System.currentTimeMillis();

        }
        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
            // Transform you image captured size according to the surface width and height
        }
        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
            return false;
        }
        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surface) {

            long now = System.currentTimeMillis();
            long timePause = now - start;

            // init --> press button
            // netIsLoaded --> restore multylayernetwork
            if(timePause > 2500 && init && netIsLoaded) {
                Log.e(TAG, "Time: "+String.valueOf(timePause));
                start = System.currentTimeMillis(); // change time
                new ThreadBufferedImage().execute();
            }
        }
    };

    private final CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(CameraDevice camera) {
            Log.e(TAG, "onOpened");
            cameraDevice = camera;
            createCameraPreview();
        }
        @Override
        public void onDisconnected(CameraDevice camera) {
            cameraDevice.close();
        }

        @Override
        public void onError(CameraDevice camera, int error) {
            cameraDevice.close();
            cameraDevice = null;
        }
    };

    final CameraCaptureSession.CaptureCallback captureCallbackListener = new CameraCaptureSession.CaptureCallback() {
        @Override
        public void onCaptureCompleted(CameraCaptureSession session, CaptureRequest request, TotalCaptureResult result) {
            super.onCaptureCompleted(session, request, result);
            Toast.makeText(context, "SavedOutside:" + imageToAnalyze, Toast.LENGTH_SHORT).show();
            createCameraPreview();

        }
    };

    protected void startBackgroundThread() {

        mBackgroundThread = new HandlerThread("Camera Background");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    protected void stopBackgroundThread() {
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    // immagine che viene bufferrizzata in quel momento --> da chiamare dentro un thread
    protected void createBufferedImage() {
        if(null == cameraDevice) {
            Log.e(TAG, "cameraDevice is null");
            return ;
        }

        CameraManager manager = (CameraManager) context.getSystemService(Context.CAMERA_SERVICE);
        try {

            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraDevice.getId());
            Size[] jpegSizes = null;

            if (characteristics != null) {
                jpegSizes = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP).getOutputSizes(ImageFormat.JPEG);
            }


            ImageReader reader = ImageReader.newInstance(width, height, ImageFormat.JPEG, 1);

            List<Surface> outputSurfaces = new ArrayList<Surface>(2);
            outputSurfaces.add(reader.getSurface());
            outputSurfaces.add(new Surface(textureView.getSurfaceTexture()));
            final CaptureRequest.Builder captureBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE);
            captureBuilder.addTarget(reader.getSurface());
            captureBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);

            // Orientation

            int rotation = this.context.getWindowManager().getDefaultDisplay().getRotation();
            Log.e(TAG,"Orientation: "+String.valueOf(ORIENTATIONS.get(rotation)));
            captureBuilder.set(CaptureRequest.JPEG_ORIENTATION, ORIENTATIONS.get(rotation));


            ImageReader.OnImageAvailableListener readerListener = new ImageReader.OnImageAvailableListener() {

                @Override
                public void onImageAvailable(ImageReader reader) {

                    Image image = null;

                    image = reader.acquireLatestImage();
                    // save image
                    ByteBuffer buffer = image.getPlanes()[0].getBuffer();
                    byte[] bytes = new byte[buffer.capacity()];
                    buffer.get(bytes);


                    imageToAnalyze = BitmapFactory.decodeByteArray(bytes,0,bytes.length);
                    image.close();

                    // start canny algorithm and save image
                    new CannyThread().execute();


                }

            };

            reader.setOnImageAvailableListener(readerListener, mBackgroundHandler);

            // capture completed -->
            final CameraCaptureSession.CaptureCallback captureListener = new CameraCaptureSession.CaptureCallback() {
                @Override
                public void onCaptureCompleted(CameraCaptureSession session, CaptureRequest request, TotalCaptureResult result) {
                    super.onCaptureCompleted(session, request, result);
                    createCameraPreview();

                }
            };

            cameraDevice.createCaptureSession(outputSurfaces, new CameraCaptureSession.StateCallback() {
                @Override
                public void onConfigured(CameraCaptureSession session) {
                    try {
                        session.capture(captureBuilder.build(), captureListener, mBackgroundHandler);
                    } catch (CameraAccessException e) {
                        e.printStackTrace();
                    }
                }
                @Override
                public void onConfigureFailed(CameraCaptureSession session) {
                }
            }, mBackgroundHandler);

        } catch (CameraAccessException e) {
            e.printStackTrace();
        }

    }

    protected void createCameraPreview() {

        try {

            SurfaceTexture texture = textureView.getSurfaceTexture();
            assert texture != null;
            texture.setDefaultBufferSize(imageDimension.getWidth(), imageDimension.getHeight());
            Surface surface = new Surface(texture);
            captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            captureRequestBuilder.addTarget(surface);
            cameraDevice.createCaptureSession(Arrays.asList(surface), new CameraCaptureSession.StateCallback(){
                @Override
                public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {

                    //The camera is already closed
                    if (null == cameraDevice) {
                        return;
                    }

                    cameraCaptureSessions = cameraCaptureSession;
                    updatePreview();

                }
                @Override
                public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {
                    Toast.makeText(context, "Configuration change", Toast.LENGTH_SHORT).show();
                }
            }, null);

        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    protected void openCamera() {
        CameraManager manager = (CameraManager) this.context.getSystemService(Context.CAMERA_SERVICE);
        Log.e(TAG, "is camera open");
        try {

            cameraId = manager.getCameraIdList()[0];
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            assert map != null;
            imageDimension = map.getOutputSizes(SurfaceTexture.class)[0];
            // Add permission for camera and let user grant the permission
            if (ActivityCompat.checkSelfPermission(this.context, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this.context, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this.context, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_CAMERA_PERMISSION);
                return;
            }
            manager.openCamera(cameraId, stateCallback, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
        Log.e(TAG, "openCamera X");
    }

    protected void updatePreview() {

        if(null == cameraDevice) {
            Log.e(TAG, "updatePreview error, return");
        }

        captureRequestBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);

        try {

            cameraCaptureSessions.setRepeatingRequest(captureRequestBuilder.build(), null, mBackgroundHandler);

        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    protected void closeCamera() {

        if (null != cameraDevice) {
            cameraDevice.close();
            cameraDevice = null;
        }

        if (null != imageReader) {
            imageReader.close();
            imageReader = null;
        }
    }

    protected  void setListener(){
        this.textureView.setSurfaceTextureListener(textureListener);

    }

    private class ThreadBufferedImage extends AsyncTask<Void, Void, Void> {

        @Override
        protected Void doInBackground(Void... voids) {
            createBufferedImage();
            return null;
        }

        @Override
        protected void onPostExecute(Void aVoid) {
            super.onPostExecute(aVoid);
        }
    }

    private class LoadNetworkThread extends AsyncTask<Void,Void,Void>{

        @Override
        protected Void doInBackground(Void... voids) {

            Resources r = context.getResources();

            if(!fileToSaveModel.exists()){
                try {
                    FileUtils.copyInputStreamToFile(r.openRawResource(idResources), fileToSaveModel);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            try {
                handposenet = ModelSerializer.restoreMultiLayerNetwork(fileToSaveModel);
            } catch (IOException e) {
                e.printStackTrace();
            }

            return null;

        }

        @Override
        protected void onPostExecute(Void aVoid) {
            super.onPostExecute(aVoid);
            // after that the net is loaded
            netIsLoaded = true;
            Log.e(TAG,"NET LOADED");
        }
    }

    // after save image it possible to read image and try to classify a image
    private class CannyThread extends AsyncTask<Void,Void,Void>{

        @Override
        protected Void doInBackground(Void... voids) {

            try {

                FileOutputStream fos = new FileOutputStream(Environment.getExternalStorageDirectory() + File.separator + nameFile);


                if (choose == 1) {
                    CannyEdgeDetector detector = new CannyEdgeDetector(CameraHandler.this);

                    //detector.setLowThreshold(0.5f);
                    //detector.setHighThreshold(1f);
                    //apply it to an image
                    detector.setSourceImage(imageToAnalyze);
                    detector.process();
                    Bitmap edges = detector.getEdgesImage();
                    Bitmap resized = Bitmap.createScaledBitmap(edges,150,150,false);
                    resized.compress(Bitmap.CompressFormat.JPEG,100,fos);
                }else {
                    Canny canny = new Canny();
                    imageToAnalyze = Bitmap.createScaledBitmap(imageToAnalyze, 438, 714, false);
                    Bitmap cannyBitmap = canny.process(imageToAnalyze);
                    Bitmap resized = Bitmap.createScaledBitmap(cannyBitmap, 150, 150, false);
                    Log.e(TAG, "Width: " + cannyBitmap.getWidth() + ", height: " + cannyBitmap.getHeight());
                    resized.compress(Bitmap.CompressFormat.JPEG, 100, fos);
                }
                fos.close();

            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }

            return null;

        }

        @Override
        protected void onPostExecute(Void aVoid) {
            super.onPostExecute(aVoid);
            Log.e(TAG,"SAVED");
            new ClassifyImageThread().execute();

        }
    }

    private class ClassifyImageThread extends AsyncTask<Void,Void,Void>{

        private String type;
        private float probability;
        @Override
        protected Void doInBackground(Void... voids) {
            Log.e(TAG,"Start Classify Image");
            AndroidNativeImageLoader loader = new AndroidNativeImageLoader(height, width, channel);
            File imageToClassify = new File(Environment.getExternalStorageDirectory()+File.separator+nameFile);
            DataNormalization scalar = new VGG16ImagePreProcessor();

            INDArray image = null;
            try {
                image = loader.asMatrix(imageToClassify);

            } catch (IOException e) {
                e.printStackTrace();
            }
            scalar.transform(image);

            INDArray output = handposenet.output(image);
            type = label.get(output.argMax(1).getInt(0));
            probability = output.getFloat(output.argMax(1).getInt(0)); //probability max

            Log.e(TAG,"Number is: "+type+", with: "+String.valueOf(probability));

            return null;
        }

        @Override
        protected void onPostExecute(Void aVoid) {
            super.onPostExecute(aVoid);
            Toast.makeText(context,type + " "+String.valueOf(probability),Toast.LENGTH_LONG ).show();
            init = false;
        }
    }

    public int getWidth(){
        return this.width;
    }

    public Context getContext(){
        return this.context;
    }
}
