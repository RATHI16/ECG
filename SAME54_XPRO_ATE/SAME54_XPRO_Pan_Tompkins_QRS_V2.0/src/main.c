/*******************************************************************************
 * ECG DSP + ML Classifier — ATSAME54P20A (Xplained Pro)
 * ======================================================
 * FULL DEMO: Pan-Tompkins QRS + BPM + ML 3-Class Classification
 *
 * Keeps ALL existing DSP (filters, QRS detection, BPM, P-wave)
 * ADDS: ML buffer → Knowledge Pack → classification output
 *
 * UART: SERCOM2 at 921600 baud (existing Data Visualizer config)
 ******************************************************************************/

#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "definitions.h"
#include "PT_algorithm.h"
#include "DV_structure.h"
#include "QRS_algorithm.h"

/* Knowledge Pack — uncomment after adding knowledgepack/ folder */
/* #include "kb.h" */

/* ============================================================
 * MODE SELECTION (same as AI Sniffer)
 * APP = 0: ML classification output → ecg_classifier.json widget
 * MDV = 1: Raw ECG streaming → .dvws workspace (waveform + BPM)
 * ============================================================ */
#define DATA_STREAMER_FORMAT_APP    0
#define DATA_STREAMER_FORMAT_MDV    1
#define DATA_STREAMER_FORMAT        DATA_STREAMER_FORMAT_APP

#define STREAM_FORMAT_IS_APP    (DATA_STREAMER_FORMAT == DATA_STREAMER_FORMAT_APP)
#define STREAM_FORMAT_IS_MDV    (DATA_STREAMER_FORMAT == DATA_STREAMER_FORMAT_MDV)

/* ============================================================
 * ML CONFIGURATION
 * ============================================================ */
#define ML_WINDOW_SIZE      2000
#define ML_NUM_CLASSES      3
#define ML_CLASS_NORMAL     0
#define ML_CLASS_AFIB       1
#define ML_CLASS_NOISE      2
#define ML_CONFIRM_COUNT    3

static int16_t ml_buffer[ML_WINDOW_SIZE];
static uint16_t ml_buf_idx = 0;
static uint8_t ml_confirmed_class = 0xFF;
static uint8_t class_counts[ML_NUM_CLASSES] = {0};
static uint8_t last_class = 0xFF;

/* ============================================================
 * ML INFERENCE
 * ============================================================ */
static int32_t ml_run_inference(void)
{
    /* ---- KNOWLEDGE PACK (uncomment when ready) ----
     * int32_t ret = kb_run_model(ml_buffer, ML_WINDOW_SIZE, 0);
     * if (ret >= 0) kb_reset_model(0);
     * return ret;
     */

    /* ---- PLACEHOLDER ---- */
    int32_t sum = 0, sum_sq = 0;
    int16_t max_val = ml_buffer[0], min_val = ml_buffer[0];
    for (uint16_t i = 0; i < ML_WINDOW_SIZE; i++) {
        int32_t v = ml_buffer[i];
        sum += v; sum_sq += v * v;
        if (ml_buffer[i] > max_val) max_val = ml_buffer[i];
        if (ml_buffer[i] < min_val) min_val = ml_buffer[i];
    }
    int32_t mean = sum / ML_WINDOW_SIZE;
    int32_t variance = (sum_sq / ML_WINDOW_SIZE) - (mean * mean);
    int16_t range = max_val - min_val;
    if (range < 50 || variance > 500000) return ML_CLASS_NOISE;
    else if (variance < 5000 && range > 200) return ML_CLASS_NORMAL;
    else return ML_CLASS_AFIB;
}

static void send_ml_classification(uint8_t classification)
{
    uint8_t header = 0xA5;
    uint8_t footer = 0x5A;
    uint8_t class_map[ML_NUM_CLASSES] = {0, 0, 0};
    if (classification < ML_NUM_CLASSES)
        class_map[classification] = 1u;
    SERCOM2_USART_Write(&header, 1);
    SERCOM2_USART_Write(class_map, 3);
    SERCOM2_USART_Write(&footer, 1);
}

/* ============================================================
 * EXISTING CODE
 * ============================================================ */
#define ARRAY_SIZE(array) (sizeof(array) / sizeof(*array))

typedef struct {
    uint8_t state;
    int16_t *data_array;
    uint32_t data_length;
    uint32_t data_index;
} sample_data_algorithm_control_t;

sample_data_algorithm_control_t sample_data_algorithm_control = {
    .state = 0, .data_array = NULL, .data_length = 0, .data_index = 0
};

static volatile bool timer_event = false;
void TC0_Interupt_Callback_1000Hz(TC_TIMER_STATUS status, uintptr_t context)
{
    timer_event = true;
}

void sample_adc_data(data_visualiser_data_t *data_visualizer_data)
{
    if (sample_data_algorithm_control.data_array == NULL) {
        while (!ADC0_ConversionStatusGet()) { }
        data_visualizer_data->new_ecg_sample = (int16_t)ADC0_ConversionResultGet();
    } else {
        int16_t sample = sample_data_algorithm_control.data_array[sample_data_algorithm_control.data_index++];
        if (sample_data_algorithm_control.data_index >= sample_data_algorithm_control.data_length)
            sample_data_algorithm_control.data_index = 0;
        data_visualizer_data->new_ecg_sample = sample;
    }
}

static int filterToUse = 0;

/* ============================================================
 * MAIN
 * ============================================================ */
int main(void)
{
    SYS_Initialize(NULL);

    TC0_TimerCallbackRegister(TC0_Interupt_Callback_1000Hz, (uintptr_t)NULL);
    TC0_TimerStart();
    ADC0_Enable();
    SERCOM2_USART_Enable();

    /* kb_model_init(); */

    memset(ml_buffer, 0, sizeof(ml_buffer));

    static data_visualiser_data_t data_visualizer_data;
    data_visualizer_data.START_OF_FRAME = 0x03;
    data_visualizer_data.END_OF_FRAME = 0xFC;
    data_visualizer_data.bpm_state = 0;

    while (true)
    {
        if (SERCOM2_USART_ReadCountGet()) {
            SERCOM2_USART_Read((uint8_t*)&filterToUse, 1);
        }

        if (timer_event)
        {
            /* ---- ADC ---- */
            sample_adc_data(&data_visualizer_data);

            /* ---- DSP: Full Pan-Tompkins Pipeline ---- */
            data_visualizer_data.new_ecg_sample_offset_corrected =
                offset_filter(data_visualizer_data.new_ecg_sample);

            int filter_output_50Hz = lp_FIR_filter_50Hz(data_visualizer_data.new_ecg_sample_offset_corrected);
            int filter_output_60Hz = lp_FIR_filter_60Hz(data_visualizer_data.new_ecg_sample_offset_corrected);

            switch (filterToUse) {
                case 1:  data_visualizer_data.lp_FIR_sample = filter_output_50Hz; break;
                case 2:  data_visualizer_data.lp_FIR_sample = filter_output_60Hz; break;
                default: data_visualizer_data.lp_FIR_sample = data_visualizer_data.new_ecg_sample_offset_corrected; break;
            }

            data_visualizer_data.mV_Conversion =
                (int16_t)((data_visualizer_data.lp_FIR_sample * 33000) / 20480);

            insert_data_to_p_wave_buffer(data_visualizer_data.lp_FIR_sample);

            data_visualizer_data.differentiated_sample =
                moving_window_differentiate(data_visualizer_data.lp_FIR_sample);
            data_visualizer_data.squared_sample =
                square_wave(data_visualizer_data.differentiated_sample);
            data_visualizer_data.integrated_sample =
                moving_window_integration(data_visualizer_data.squared_sample);
            data_visualizer_data.pt_threshold_signal =
                pt_signal_max_filter(data_visualizer_data.integrated_sample);

            detect_qrs(&data_visualizer_data);

            data_visualizer_data.bpm_average = get_bpm_average();
            data_visualizer_data.bpm_min = get_bpm_min();
            data_visualizer_data.bpm_max = get_bpm_max();

            read_p_wave_buffer_to_output(&data_visualizer_data);

            /* ---- APP MODE: ML Classification ---- */
#if STREAM_FORMAT_IS_APP
            ml_buffer[ml_buf_idx++] = data_visualizer_data.lp_FIR_sample;

            if (ml_buf_idx >= ML_WINDOW_SIZE)
            {
                int32_t result = ml_run_inference();
                if (result >= 0 && result < ML_NUM_CLASSES)
                {
                    if ((uint8_t)result == last_class) {
                        class_counts[result]++;
                        if (class_counts[result] >= ML_CONFIRM_COUNT) {
                            ml_confirmed_class = (uint8_t)result;
                            send_ml_classification(ml_confirmed_class);
                            memset(class_counts, 0, sizeof(class_counts));
                        }
                    } else {
                        memset(class_counts, 0, sizeof(class_counts));
                        class_counts[result] = 1;
                        last_class = (uint8_t)result;
                    }
                }
                ml_buf_idx = 0;
            }
#endif

            /* ---- MDV MODE: Stream full DV packet (waveform + BPM) ---- */
#if STREAM_FORMAT_IS_MDV
            SERCOM2_USART_Write((uint8_t *)&data_visualizer_data, sizeof(data_visualizer_data));
#endif

            timer_event = false;
        }

        SYS_Tasks();
    }

    return (EXIT_FAILURE);
}
