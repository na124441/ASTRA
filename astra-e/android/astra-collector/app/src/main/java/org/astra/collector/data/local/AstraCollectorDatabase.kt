package org.astra.collector.data.local

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(
    entities = [TaskEntity::class, UploadRecordEntity::class, DeviceEntity::class],
    version = 1,
    exportSchema = false
)
abstract class AstraCollectorDatabase : RoomDatabase() {
    abstract fun taskDao(): TaskDao
    abstract fun uploadDao(): UploadDao

    companion object {
        @Volatile
        private var INSTANCE: AstraCollectorDatabase? = null

        fun getInstance(context: Context): AstraCollectorDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AstraCollectorDatabase::class.java,
                    "astra_collector.db"
                ).fallbackToDestructiveMigration().build()
                INSTANCE = instance
                instance
            }
        }
    }
}
