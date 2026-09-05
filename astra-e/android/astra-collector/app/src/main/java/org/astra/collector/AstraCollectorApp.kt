package org.astra.collector

import android.app.Application
import android.util.Log

class AstraCollectorApp : Application() {
    override fun onCreate() {
        super.onCreate()
        Log.i("AstraCollectorApp", "ASTRA Collector initialized")
    }
}
