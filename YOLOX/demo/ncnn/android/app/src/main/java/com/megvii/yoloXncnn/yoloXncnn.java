// Copyright (C) Megvii, Inc. and its affiliates. All rights reserved.

package com.megvii.yoloXncnn;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class YOLOXncnn
{
    public native boolean Init(AssetManager mgr);

    public class Obj
    {
        public float x;
        public float y;
        public float w;
        public float h;
        public String label;
        public float prob;
    }

    public native Obj[] Detect(Bitmap bitmap, boolean use_gpu);

    static {
        System.loadLibrary("yoloXncnn");
    }
}
