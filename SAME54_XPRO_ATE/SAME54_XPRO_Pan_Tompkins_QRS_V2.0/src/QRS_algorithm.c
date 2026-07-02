/* ************************************************************************** */
/** Descriptive File Name

  @Company
    Company Name

  @File Name
    filename.c

  @Summary
    Brief description of the file.

  @Description
    Describe the purpose of this file.
 */
/* ************************************************************************** */

/* ************************************************************************** */
/* ************************************************************************** */
/* Section: Included Files                                                    */
/* ************************************************************************** */
/* ************************************************************************** */

/* This section lists the other files that are included in this file.
 */
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>

#include "QRS_algorithm.h"
#include "PT_algorithm.h"

//should detect to 10 bpm
#define SAMPLE_RATE_PER_SECOND 1000
#define PT_OUTPUT_BUFFER_LENGTH     (16384)
#define PT_OUTPUT_SAMPLES_PER_MINUTE      (60 * SAMPLE_RATE_PER_SECOND)
#define PT_OUTPUT_SAMPLES_PER_MINUTE_WITH_ROUNDING      (PT_OUTPUT_SAMPLES_PER_MINUTE + (SAMPLE_RATE_PER_SECOND / 2))
static volatile int pt_input_buffer[PT_OUTPUT_BUFFER_LENGTH];
static int32_t pt_index = 0;

int32_t pt_previous_rising_edge = 0;
int32_t pt_previous_peak = 0;
int32_t pt_previous_falling_edge = 0;

#define BPM_BUFFER_LENGTH   (16)
static int bpm_buffer[BPM_BUFFER_LENGTH];
static int16_t bpm_buffer_index = 0;
static int16_t bpm_buffer_count;
static int16_t bpm_buffer_sum;

//should go down to 10 bpm ~ 6 seconds data
#define P_WAVE_BUFFER_LENGTH    (16384)
static int16_t p_wave_buffer[P_WAVE_BUFFER_LENGTH];

static int16_t p_wave_index_to_write = 0;
static int16_t p_wave_index_to_read = 0;
static int16_t p_wave_count_to_read = 0;

#define BASE_LINE_NORMAL    2

#define QRS_RISING_EDGE_DELTA_X     15
#define QRS_PEAK_DELTA_X            10
#define QRS_FALLING_EDGE_DELTA_X    5
#define QRS_TIMEOUT                 300
#define QRS_TIMEOUT_MULIPLIER       1


typedef enum BPM_STATE
{
    BPM_PRIMING_BUFFER,
    BPM_FINDING_RISING_EDGE,
    BPM_FINDING_PEAK,
    BPM_FINDING_FALLING_EDGE,
    BPM_TIMEOUT
}BPM_STATE_E;

typedef enum P_WAVE_STATE
{
    P_WAVE_STATE_PRIME,
    P_WAVE_STATE_FINDING_CROSSING,
    P_WAVE_STATE_SCROLL_CROSSING,
}P_WAVE_STATE_E;

static int get_feature_count_difference(int current_feature_count, int previous_feature_count)
{    
    int difference = 0; 
    if(current_feature_count >= previous_feature_count)
    {
        difference = current_feature_count - previous_feature_count;
    }
    else
    {
        difference = (PT_OUTPUT_BUFFER_LENGTH - previous_feature_count) + current_feature_count;
    }
    return difference;
    //(current_feature_count - previous_feature_count) % QRS_BUFFER_LENGTH;
}

int16_t get_bpm_min()
{
    int16_t bpm_min = INT16_MAX;
    if(bpm_buffer_count != 0)
    {
        for(int i = 0 ; i < BPM_BUFFER_LENGTH ; i++)
        {
            if(bpm_buffer[i] != 0 &&
                    bpm_buffer[i] < bpm_min)
            {
                bpm_min = bpm_buffer[i];
            }
        }
    }
    return bpm_min;
}

int16_t get_bpm_max()
{
    int16_t bpm_max = 0;
    if(bpm_buffer_count != 0)
    {
        for(int i = 0; i < BPM_BUFFER_LENGTH ; i++)
        {
            if(bpm_buffer[i] != 0 &&
                    bpm_buffer[i] > bpm_max)
            {
                bpm_max = bpm_buffer[i];
            }
        }
    }
    return bpm_max;
}

int16_t get_bpm_average()
{
    if(bpm_buffer_count != 0)
    {
        return (bpm_buffer_sum + (BPM_BUFFER_LENGTH / 2)) / bpm_buffer_count;
    }
    return 0;
}

static void insert_bpm(int bpm)
{
    
    if(bpm_buffer_count >= BPM_BUFFER_LENGTH)
    {
        bpm_buffer_sum -= bpm_buffer[bpm_buffer_index];        
    }
    else
    {
        bpm_buffer_count++;
    }
    
    bpm_buffer[bpm_buffer_index] = bpm;
    bpm_buffer_sum += bpm;
    
    bpm_buffer_index = (bpm_buffer_index + 1) % BPM_BUFFER_LENGTH;
    
}

static int get_bpm(int feature_count_difference)
{
    int bpm = 0;
    if(feature_count_difference > 0)
    {
        bpm = PT_OUTPUT_SAMPLES_PER_MINUTE_WITH_ROUNDING / feature_count_difference;
    }
    else
    {
        bpm = -1;
    }
    insert_bpm(bpm);
    return bpm;    
}

static int get_index_back_in_time(int index, int delta)
{
    if(index >= delta)
    {
        return index - delta;
    }
    return (PT_OUTPUT_BUFFER_LENGTH - delta) + index;
}

static int get_index_forward_in_time(int index, int delta)
{
    int to_return = index + delta;
    if(to_return >= PT_OUTPUT_BUFFER_LENGTH)
    {
        to_return -= PT_OUTPUT_BUFFER_LENGTH;
    }
    return to_return;
}

void detect_qrs(data_visualiser_data_t *data_packet)
{
    // Implement the QRS detection logic
    // This would involve setting a threshold and identifying peaks that cross the threshold
    // This is a complex process and is left as an exercise
    
    int32_t input_sample = data_packet->integrated_sample;
    int16_t qrs_width = 0;
    
    pt_input_buffer[pt_index] = input_sample;
//    static int bpm = 0;
    static int bpm_timeout = 0;
    uint16_t qrs_previous_index;
    uint16_t qrs_previous_previous_index;
    
    //if waiting for next bpm is longer than last peak peak interval start scaling bpm down
    if(data_packet->R2R_interval != 0)
    {
        int current_count_between_features = get_feature_count_difference(pt_index, pt_previous_peak);
        if(current_count_between_features > data_packet->R2R_interval)
        {
            data_packet->bpm = get_bpm(current_count_between_features);
        }
    }

    switch(data_packet->bpm_state)
    {
        case BPM_PRIMING_BUFFER:
            //do nothing, buffer not yet full
            if(pt_index > QRS_TIMEOUT)
            {
                data_packet->bpm_state = BPM_FINDING_RISING_EDGE;
            }
            break;
        case BPM_FINDING_RISING_EDGE:
            qrs_previous_index = get_index_back_in_time(pt_index, QRS_RISING_EDGE_DELTA_X);
            
            if((pt_input_buffer[pt_index] - pt_input_buffer[qrs_previous_index]) > (int32_t)((data_packet->pt_threshold_signal) >> 2 ))
            {        
//                data_packet->count_between_peaks = get_feature_count_difference(qrs_index, qrs_previous_rising_edge);
//                data_packet->bpm = get_bpm(data_packet->count_between_peaks);
                pt_previous_rising_edge = qrs_previous_index;
                
                //find point where deviates from zero
                
                data_packet->bpm_state = BPM_FINDING_PEAK;
            }
            break;
        case BPM_FINDING_PEAK:            
            qrs_previous_index = get_index_back_in_time(pt_index, QRS_PEAK_DELTA_X);
            qrs_previous_previous_index = get_index_back_in_time(qrs_previous_index, QRS_PEAK_DELTA_X);
            
            if(pt_input_buffer[qrs_previous_index] > pt_input_buffer[pt_index] &&
                    pt_input_buffer[qrs_previous_index] > pt_input_buffer[qrs_previous_previous_index])
            {
                bool peak_found = true;
                //check middle sample is largest                
                for (int i = 1 ; i < QRS_PEAK_DELTA_X ; i++ )
                {
                    int neg_index = get_index_back_in_time(qrs_previous_index, i);
                    int pos_index = get_index_forward_in_time(qrs_previous_index, i);
                    
                    //if peak is less than waveforms either side break out;
                    if(pt_input_buffer[qrs_previous_index] < pt_input_buffer[neg_index] ||
                            pt_input_buffer[qrs_previous_index] < pt_input_buffer[pos_index])
                    {
                        // not the peak
                        peak_found = false;
                        break;
                    }
                }
                if(peak_found)
                {
                    //all data points are less than data at 'qrs_previous_index'
                    data_packet->R2R_interval = get_feature_count_difference(pt_index, pt_previous_peak);
                    
                    //count to read either 3/8 or 4/8 of gap between peaks
                    p_wave_count_to_read = (3 * (data_packet->R2R_interval >> 3));
                    p_wave_index_to_read = get_index_back_in_time(pt_index, p_wave_count_to_read + MWI_WINDOW_SIZE + (MWD_WINDOW_SIZE << 1) );//get_index_back_in_time(pt_index, ((data_packet->count_between_peaks * 6) >> 3)  + MWI_WINDOW_SIZE + MWD_WINDOW_SIZE);
                    
                    //get min, max and 70% point between min and max, find number of continuous regions above line
                    int min = INT16_MAX;
                    int max = INT16_MIN;
                    int temp_count_to_process = p_wave_count_to_read;
                    int temp_index_to_process = p_wave_index_to_read;
                    
                    while(temp_count_to_process)
                    {
                        if(p_wave_buffer[temp_index_to_process] > max)
                        {
                            max = p_wave_buffer[temp_index_to_process];
                        }
                        if(p_wave_buffer[temp_index_to_process] < min)
                        {
                            min = p_wave_buffer[temp_index_to_process];
                        }
                        temp_count_to_process--;
                        temp_index_to_process = (temp_index_to_process + 1) % P_WAVE_BUFFER_LENGTH;
                    }
                    
                    int temp_diff = max - min;
                    int temp_threshold = min + ((8 * temp_diff) / 10);
                    data_packet->p_wave_threshold = temp_threshold;
                    
                    
                    //find crossings
                    temp_count_to_process = p_wave_count_to_read;
                    temp_index_to_process = p_wave_index_to_read;
                    uint8_t crossing_count = 0;
                    P_WAVE_STATE_E p_wave_state = P_WAVE_STATE_PRIME;
                    
                    while(temp_count_to_process)
                    {
                        switch(p_wave_state)
                        {
                            case P_WAVE_STATE_PRIME:
                                //wait for signal to be below threshold
                                if(p_wave_buffer[temp_index_to_process] > temp_threshold)
                                {
                                }
                                else
                                {
                                    p_wave_state = P_WAVE_STATE_FINDING_CROSSING;
                                }                                
                                break;
                            case P_WAVE_STATE_FINDING_CROSSING:
                                //if threshold crossed then log
                                if(p_wave_buffer[temp_index_to_process] > temp_threshold)
                                {
                                    crossing_count++;
                                    p_wave_state = P_WAVE_STATE_SCROLL_CROSSING;
                                }
                                else
                                {
                                }
                                break;
                            case P_WAVE_STATE_SCROLL_CROSSING:
                                //wait for signal to go below threshold
                                if(p_wave_buffer[temp_index_to_process] > temp_threshold)
                                {                                    
                                }
                                else
                                {
                                    p_wave_state = P_WAVE_STATE_FINDING_CROSSING;
                                }
                                break;                            
                        }
                        
                        temp_count_to_process--;
                        temp_index_to_process = (temp_index_to_process + 1) % P_WAVE_BUFFER_LENGTH;
                    }
                    data_packet->p_wave_crossings = crossing_count + 1;
                    
                    if(data_packet->p_wave_crossings != BASE_LINE_NORMAL)
                    {
                        data_packet->p_wave_abnormal = 1;
                    }
                    else
                    {
                        data_packet->p_wave_ok = 1;                        
                    }
                    
                    data_packet->bpm = get_bpm(data_packet->R2R_interval);

                    pt_previous_peak = pt_index;
                    data_packet->bpm_state = BPM_FINDING_FALLING_EDGE;
                }
            }
            break;
        case BPM_FINDING_FALLING_EDGE:
            qrs_previous_index = get_index_back_in_time(pt_index, QRS_FALLING_EDGE_DELTA_X);
            if(pt_input_buffer[pt_index] < (data_packet->pt_threshold_signal >> 3))
            {
                if(pt_input_buffer[qrs_previous_index] - pt_input_buffer[pt_index] < (data_packet->pt_threshold_signal >> 4))
                {
    //                data_packet->count_between_peaks = get_feature_count_difference(qrs_index, qrs_previous_falling_edge);
    //                data_packet->bpm = get_bpm(data_packet->count_between_peaks);

                    pt_previous_falling_edge = pt_index;
                    data_packet->bpm_state = BPM_TIMEOUT;
                    qrs_width = get_feature_count_difference(pt_previous_falling_edge, pt_previous_rising_edge);
                    bpm_timeout = qrs_width * QRS_TIMEOUT_MULIPLIER;

                    data_packet->qrs_width = qrs_width;
                }
            }
            break;
        case BPM_TIMEOUT:
            bpm_timeout--;
            if(bpm_timeout < 0)
            {
                bpm_timeout = 0;
                
                data_packet->bpm_state = BPM_FINDING_RISING_EDGE;
            }
            break;
    }
    
    //can be optimised for (pt_index + 1) & (PT_OUTPUT_BUFFER_LENGTH - 1)
    pt_index = ((pt_index + 1) % PT_OUTPUT_BUFFER_LENGTH);
}



void insert_data_to_p_wave_buffer(int16_t input_sample)
{
    p_wave_buffer[p_wave_index_to_write] = input_sample;
    
    //can be optimised for (p_wave_index_to_write + 1) & (PT_OUTPUT_BUFFER_LENGTH - 1)
    p_wave_index_to_write = ((p_wave_index_to_write + 1) % P_WAVE_BUFFER_LENGTH);
}

void read_p_wave_buffer_to_output(data_visualiser_data_t *data_packet)
{
    if(p_wave_count_to_read)
    {
        data_packet->p_wave_data = p_wave_buffer[p_wave_index_to_read];
        
        p_wave_count_to_read--;
        //can be optimised for (p_wave_index_to_write + 1) & (PT_OUTPUT_BUFFER_LENGTH - 1)
        p_wave_index_to_read = ((p_wave_index_to_read + 1) % P_WAVE_BUFFER_LENGTH);        
    }
    else
    {
        data_packet->p_wave_data = 0;
        data_packet->p_wave_threshold = 0;
        data_packet->p_wave_crossings = 0;
        data_packet->p_wave_abnormal = 0;
        data_packet->p_wave_ok = 0; 
    }
}

/* *****************************************************************************
 End of File
 */
