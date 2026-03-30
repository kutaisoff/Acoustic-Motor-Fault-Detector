/*
 * Motor Fault Detector — ESP32-WROOM-32 (без PSRAM) + INMP441 + TFLite Micro
 * Запуск записи осуществляется вручную через Serial Monitor!
 */

#include <Arduino.h>
#include <driver/i2s.h>
#include "arduinoFFT.h"
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "motor_fault_model.h"
#include "normalization.h"

// ======================== Пины ========================
#define I2S_WS    26
#define I2S_SD    33
#define I2S_SCK   25
#define I2S_PORT  I2S_NUM_0

// ======================== Параметры ========================
const int I2S_SAMPLE_RATE = 16000;
const int RECORD_SAMPLES  = 102400;  // 6.4 сек при 16000 Гц

const int NUM_MFCC   = 20;
const int NUM_FILT   = 40;
const int FFT_SIZE   = 1024;
const int WIN_LEN    = 400;
const int STEP_LEN   = 160;
const int HALF_FFT   = FFT_SIZE / 2 + 1;
const int NUM_FEATURES = 80;
const int NUM_WINDOWS  = (RECORD_SAMPLES - WIN_LEN) / STEP_LEN + 1;
const int LIFTER_L     = 22;

// ======================== TFLite ========================
const int kTensorArenaSize = 40 * 1024;
__attribute__((aligned(16))) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* tfl_interpreter = nullptr;
TfLiteTensor* input_tensor  = nullptr;
TfLiteTensor* output_tensor = nullptr;

class DebugReporter : public tflite::ErrorReporter {
public:
  int Report(const char* format, va_list args) override {
    char buf[256];
    vsnprintf(buf, sizeof(buf), format, args);
    Serial.print("[TFL] ");
    Serial.println(buf);
    return 0;
  }
};
DebugReporter reporter;

// ======================== FFT ========================
arduinoFFT FFT = arduinoFFT();
double vReal[FFT_SIZE];
double vImag[FFT_SIZE];
int16_t win_buf[WIN_LEN];

// ======================== Таблицы ========================
int mel_bins[NUM_FILT + 2];      
float lifter[NUM_MFCC];          

float hz2mel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }
float mel2hz(float m)  { return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f); }

void init_tables() {
    float lo = hz2mel(0.0f);
    float hi = hz2mel(8000.0f); 
    for (int i = 0; i < NUM_FILT + 2; i++) {
        float mel = lo + i * (hi - lo) / (NUM_FILT + 1);
        mel_bins[i] = (int)floorf((FFT_SIZE + 1) * mel2hz(mel) / 16000.0f);
    }
    for (int n = 0; n < NUM_MFCC; n++)
        lifter[n] = 1.0f + (LIFTER_L / 2.0f) * sinf(M_PI * n / LIFTER_L);
}

// ======================== MFCC ========================
void compute_mfcc(float* frame, float* out) {
    float energy = 0.0f;
    for (int i = 0; i < WIN_LEN; i++) {
        energy += frame[i] * frame[i];
    }
    if (energy < 1e-10f) energy = 1e-10f;

    for (int i = WIN_LEN - 1; i > 0; i--)
        frame[i] -= 0.97f * frame[i - 1];

    for (int i = 0; i < FFT_SIZE; i++) {
        vReal[i] = (i < WIN_LEN) ? (double)frame[i] : 0.0;
        vImag[i] = 0.0;
    }
    FFT.Compute(vReal, vImag, FFT_SIZE, FFT_FORWARD);

    float pspec[HALF_FFT];
    for (int i = 0; i < HALF_FFT; i++) {
        pspec[i] = (float)(vReal[i]*vReal[i] + vImag[i]*vImag[i]) / FFT_SIZE;
    }

    float mel_e[NUM_FILT];
    for (int m = 0; m < NUM_FILT; m++) {
        int s = mel_bins[m], mid = mel_bins[m+1], e = mel_bins[m+2];
        float sum = 0.0f;
        if (mid > s) {
            for (int k = s; k < mid && k < HALF_FFT; k++)
                sum += pspec[k] * (float)(k - s) / (mid - s);
        }
        if (e > mid) {
            for (int k = mid; k <= e && k < HALF_FFT; k++)
                sum += pspec[k] * (float)(e - k) / (e - mid);
        }
        if (sum < 1e-10f) sum = 1e-10f;
        mel_e[m] = logf(sum);
    }

    float cf = sqrtf(2.0f / NUM_FILT);
    for (int k = 0; k < NUM_MFCC; k++) {
        float s = 0.0f;
        for (int m = 0; m < NUM_FILT; m++)
            s += mel_e[m] * cosf(M_PI * k * (m + 0.5f) / NUM_FILT);
        out[k] = cf * s;
    }

    for (int n = 0; n < NUM_MFCC; n++) out[n] *= lifter[n];

    out[0] = logf(energy);
}

// ======================== I2S ========================
bool initI2S() {
    i2s_config_t cfg = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = (uint32_t)I2S_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 512,
        .use_apll = false
    };
    i2s_pin_config_t pins = {
        .bck_io_num = I2S_SCK, .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE, .data_in_num = I2S_SD
    };
    if (i2s_driver_install(I2S_PORT, &cfg, 0, NULL) != ESP_OK) return false;
    if (i2s_set_pin(I2S_PORT, &pins) != ESP_OK) return false;
    i2s_zero_dma_buffer(I2S_PORT);
    return true;
}

void readI2S(int16_t* dst, int count) {
    int32_t tmp[256];
    size_t br;
    int got = 0;
    while (got < count) {
        int n = min(256, count - got);
        i2s_read(I2S_PORT, tmp, n * sizeof(int32_t), &br, portMAX_DELAY);
        int s = br / sizeof(int32_t);
        for (int i = 0; i < s && got < count; i++)
            dst[got++] = (int16_t)((tmp[i] >> 8) >> 8);
    }
}

// ======================== Потоковая обработка ========================
void processAudio(float* features) {
    float sum[NUM_MFCC] = {0}, sum2[NUM_MFCC] = {0};
    float mx[NUM_MFCC], mn[NUM_MFCC];
    for (int i = 0; i < NUM_MFCC; i++) { mx[i] = -1e9f; mn[i] = 1e9f; }

    readI2S(win_buf, WIN_LEN);
    float frame[WIN_LEN], mfcc[NUM_MFCC];
    float gmax = 0.0f;

    Serial.println("Слушаю мотор (6.4с)...");

    for (int w = 0; w < NUM_WINDOWS; w++) {
        // Очистка от аппаратного шума микрофона (DC Offset)
        float mean_val = 0.0f;
        for (int i = 0; i < WIN_LEN; i++) {
            mean_val += (float)win_buf[i];
        }
        mean_val /= WIN_LEN;

        for (int i = 0; i < WIN_LEN; i++) {
            float v = (float)win_buf[i] - mean_val; // Центрируем сигнал
            frame[i] = v;
            float a = fabsf(v);
            if (a > gmax) gmax = a;
        }

        compute_mfcc(frame, mfcc);

        for (int i = 0; i < NUM_MFCC; i++) {
            float v = mfcc[i];
            sum[i] += v; sum2[i] += v*v;
            if (v > mx[i]) mx[i] = v;
            if (v < mn[i]) mn[i] = v;
        }

        memmove(win_buf, win_buf + STEP_LEN, (WIN_LEN - STEP_LEN) * sizeof(int16_t));
        readI2S(&win_buf[WIN_LEN - STEP_LEN], STEP_LEN);
    }

    if (gmax < 1e-6f) gmax = 1e-6f;
    float eshift = 2.0f * logf(gmax);

    int idx = 0;
    for (int i = 0; i < NUM_MFCC; i++) {
        float m = sum[i] / NUM_WINDOWS;
        features[idx++] = (i == 0) ? m - eshift : m;
    }
    for (int i = 0; i < NUM_MFCC; i++) {
        float m = sum[i] / NUM_WINDOWS;
        float var = sum2[i] / NUM_WINDOWS - m*m;
        features[idx++] = (var > 0) ? sqrtf(var) : 0.0f;
    }
    for (int i = 0; i < NUM_MFCC; i++)
        features[idx++] = (i == 0) ? mx[i] - eshift : mx[i];
    for (int i = 0; i < NUM_MFCC; i++)
        features[idx++] = (i == 0) ? mn[i] - eshift : mn[i];

    for (int i = 0; i < NUM_FEATURES; i++)
        features[i] = (features[i] - MEAN[i]) / (STD[i] + 1e-6f);
}

// ======================== Setup ========================
void setup() {
    Serial.begin(115200);
    delay(2000);

    Serial.println("\n=============================");
    Serial.println("=== Motor Fault Detector ===");
    Serial.println("=============================");

    if (!initI2S()) { Serial.println("I2S FAIL"); while(1) delay(1000); }
    init_tables();

    tfl_model = tflite::GetModel(motor_fault_model_tflite);
    if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model version FAIL"); while(1) delay(1000);
    }
    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter interp(
        tfl_model, resolver, tensor_arena, kTensorArenaSize, &reporter, nullptr, nullptr);
    tfl_interpreter = &interp;
    if (tfl_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors FAIL"); while(1) delay(1000);
    }
    input_tensor  = tfl_interpreter->input(0);
    output_tensor = tfl_interpreter->output(0);
    Serial.println("=== READY ===\n");
}

// ======================== Loop ========================
void loop() {
    Serial.println("\n[ Ожидание команды... Введи любой символ в монитор порта и нажми Enter для старта записи ]");
    
    // Ждем символ от пользователя
    while (!Serial.available()) {
        delay(100);
    }
    
    // Очищаем буфер
    while (Serial.available()) {
        Serial.read();
        delay(5);
    }

    float features[NUM_FEATURES];
    processAudio(features);

    for (int i = 0; i < NUM_FEATURES; i++) {
        int32_t q = (int32_t)roundf(features[i] / input_tensor->params.scale
                                    + input_tensor->params.zero_point);
        input_tensor->data.int8[i] = (int8_t)constrain(q, -128, 127);
    }

    if (tfl_interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke error!"); return;
    }

    const char* names[] = {"Healthy","Bearing Fault","Bow","Broken Bars","Misalignment","Unbalance"};
    float probs[6]; float best = -1e9f; int cls = 0;
    for (int i = 0; i < 6; i++) {
        probs[i] = (output_tensor->data.int8[i] - output_tensor->params.zero_point)
                   * output_tensor->params.scale;
        if (probs[i] > best) { best = probs[i]; cls = i; }
    }

    Serial.println("========================================");
    Serial.printf("  RESULT: %s (%.1f%%)\n", names[cls], best * 100.0f);
    for (int i = 0; i < 6; i++)
        Serial.printf("    %s: %.1f%%%s\n", names[i], probs[i]*100, i==cls?" <<<":"");
    Serial.println("========================================\n");
}