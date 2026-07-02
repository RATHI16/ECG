
#include <stdint.h>

#include "peripheral/adc/plib_adc0.h"
#include "PT_algorithm.h"
#include <math.h>


// Moving window integration state
int mwi_stream[MWI_WINDOW_SIZE] = {0};
int mwi_index = 0;

// Moving window differential state
int mwd_stream[MWD_WINDOW_SIZE] = {0};
int mwd_index = 0;
int mwd_index_previous = 0;

// Example window size for Averaging
//for 50 Hz
#define MWA_WINDOW_SIZE_50HZ (20) 
//for 60 Hz
#define MWA_WINDOW_SIZE_60HZ (16) 

// Moving window Average state
int16_t mwa_stream_50Hz[MWA_WINDOW_SIZE_50HZ] = {0};
int16_t mwa_stream_60Hz[MWA_WINDOW_SIZE_60HZ] = {0};
int mwa_index_50Hz = 0;
int mwa_index_60Hz = 0;


// Implement the functions here
int16_t offset_filter(int16_t input_sample)
{    
#if 0   //enable filter
#define MAX_FILTER_SAMPLES  (2048)
#define MAX_FILTER_BITS     (11)
    static int16_t offset_data[MAX_FILTER_SAMPLES] = {0};
    static int16_t offset_index = 0;
    static int32_t offset_sum = 0;
    
    offset_sum -= offset_data[offset_index];
    offset_data[offset_index] = input_sample;    
    offset_sum += input_sample;
    
    offset_index++;
    offset_index &= (MAX_FILTER_SAMPLES - 1);
    
    return (int16_t)(input_sample - ((offset_sum + (MAX_FILTER_SAMPLES >> 1)) >> MAX_FILTER_BITS));
#else   //disable filter
    
    return input_sample;
    
#endif    
}

int16_t lp_FIR_filter_50Hz(int16_t input_sample)
{    
    mwa_stream_50Hz[mwa_index_50Hz] = input_sample;      
    int32_t mwa_result = 0;
    
    //can be optimised to subtract most historical sample and add new sample
    for(int i = 0 ; i < MWA_WINDOW_SIZE_50HZ ; i++)
    {
        int index = (mwa_index_50Hz + i) % MWA_WINDOW_SIZE_50HZ;
        int32_t stream_value = mwa_stream_50Hz[index];
        mwa_result += stream_value;
    }
    
    mwa_result = (mwa_result + (MWA_WINDOW_SIZE_50HZ / 2)) / MWA_WINDOW_SIZE_50HZ;
    mwa_index_50Hz = ((mwa_index_50Hz + 1) % MWA_WINDOW_SIZE_50HZ);
    return (int16_t)mwa_result;    
}

int16_t lp_FIR_filter_60Hz(int16_t input_sample)
{    
    mwa_stream_60Hz[mwa_index_60Hz] = input_sample;      
    int32_t mwa_result = 0;
    
    //can be optimised to subtract most historical sample and add new sample
    for(int i = 0 ; i < MWA_WINDOW_SIZE_60HZ ; i++)
    {
        int index = (mwa_index_60Hz + i) % MWA_WINDOW_SIZE_60HZ;
        int32_t stream_value = mwa_stream_60Hz[index];
        mwa_result += stream_value;
    }
    
    mwa_result = (mwa_result + (MWA_WINDOW_SIZE_60HZ / 2)) / MWA_WINDOW_SIZE_60HZ;
    mwa_index_60Hz = ((mwa_index_60Hz + 1) % MWA_WINDOW_SIZE_60HZ);
    return (int16_t)mwa_result;    
}

int16_t moving_window_differentiate(int16_t input_sample)
{
    
    int differential = input_sample - mwd_stream[mwd_index];
    
    // Differentiate the signal over a moving window
    mwd_stream[mwd_index] = input_sample;
    
    mwd_index = (mwd_index + 1) % MWD_WINDOW_SIZE; // Circular buffer
    
    return differential;// / MWD_WINDOW_SIZE;
}

uint32_t square_wave(int16_t input_sample)
{
    return input_sample * input_sample;
}

uint32_t moving_window_integration(uint32_t input_sample)
{
    // Integrate the signal over a moving window
    mwi_stream[mwi_index] = input_sample;
    int integration_sum = 0;
    for (int i = 0; i < MWI_WINDOW_SIZE; i++)
    {
        integration_sum += mwi_stream[i];
    }
    mwi_index = (mwi_index + 1) % MWI_WINDOW_SIZE; // Circular buffer
    return integration_sum;
}

uint32_t pt_signal_max_filter(uint32_t pt_output_sample)
{
    #define DECAY_RATE   (1)
    #define SAMPLE_RATE_PER_100_MS (100)
    #define TIMEOUT_SAMPLES (5000)

    static int16_t  count = 0;
    static int16_t  timeout = 0;
    static uint32_t  max_pt_signal = 0;
    
    if(pt_output_sample > max_pt_signal)
    {
        max_pt_signal = pt_output_sample;
        count = SAMPLE_RATE_PER_100_MS;
        timeout = TIMEOUT_SAMPLES;
    }
    else if(count)
    {
        count--;
    }
    else
    {
        count = SAMPLE_RATE_PER_100_MS;
        int32_t to_adjust = max_pt_signal >> 5;
        to_adjust++;
        
        if(max_pt_signal > to_adjust)
        {
            max_pt_signal -= to_adjust;//(1 + ((max_pt_signal * (DECAY_RATE)) / 100));
        }
    }
    
    if(timeout)
    {
        timeout--;
        if(!timeout)
        {
            max_pt_signal = pt_output_sample;
        }
    }
        
    return max_pt_signal;
}
