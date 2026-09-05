package org.astra.collector.data.remote

import okhttp3.OkHttpClient
import okhttp3.RequestBody
import okhttp3.logging.HttpLoggingInterceptor
import org.astra.collector.data.model.ChunkUploadResult
import org.astra.collector.data.model.CollectionTask
import org.astra.collector.data.model.DeviceAuthResponse
import org.astra.collector.data.model.DeviceRegisterRequest
import org.astra.collector.data.model.UploadCompleteRequest
import org.astra.collector.data.model.UploadInitiateRequest
import org.astra.collector.data.model.UploadInitiateResponse
import org.astra.collector.data.model.UploadStatusResponse
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.Header
import retrofit2.http.POST
import retrofit2.http.PUT
import retrofit2.http.Path
import retrofit2.http.Query
import java.util.concurrent.TimeUnit

interface CollectorApiService {

    @POST("api/v1/collector/auth/register")
    suspend fun registerDevice(
        @Body req: DeviceRegisterRequest
    ): DeviceAuthResponse

    @GET("api/v1/collector/tasks/next")
    suspend fun getNextTask(
        @Query("collector_id") collectorId: String
    ): CollectionTask?

    @POST("api/v1/collector/uploads/initiate")
    suspend fun initiateUpload(
        @Body req: UploadInitiateRequest
    ): UploadInitiateResponse

    @PUT("api/v1/collector/uploads/{upload_id}/chunks/{chunk_index}")
    suspend fun uploadChunk(
        @Path("upload_id") uploadId: String,
        @Path("chunk_index") chunkIndex: Int,
        @Header("X-Chunk-SHA256") chunkSha256: String?,
        @Body chunkData: RequestBody
    ): ChunkUploadResult

    @POST("api/v1/collector/uploads/{upload_id}/complete")
    suspend fun completeUpload(
        @Path("upload_id") uploadId: String,
        @Body req: UploadCompleteRequest
    ): UploadStatusResponse

    @GET("api/v1/collector/uploads/{upload_id}/status")
    suspend fun getUploadStatus(
        @Path("upload_id") uploadId: String
    ): UploadStatusResponse

    companion object {
        fun create(baseUrl: String): CollectorApiService {
            val logging = HttpLoggingInterceptor().apply {
                level = HttpLoggingInterceptor.Level.BASIC
            }

            val client = OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)
                .writeTimeout(60, TimeUnit.SECONDS)
                .addInterceptor(logging)
                .build()

            val sanitizedUrl = if (baseUrl.endsWith("/")) baseUrl else "$baseUrl/"

            return Retrofit.Builder()
                .baseUrl(sanitizedUrl)
                .client(client)
                .addConverterFactory(GsonConverterFactory.create())
                .build()
                .create(CollectorApiService::class.java)
        }
    }
}
