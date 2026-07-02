/* ************************************************************************** */
/** Descriptive File Name

  @Company
    Company Name

  @File Name
    filename.h

  @Summary
    Brief description of the file.

  @Description
    Describe the purpose of this file.
 */
/* ************************************************************************** */

#ifndef GLOBAL_STRUCTURES_H    /* Guard against multiple inclusion */
#define GLOBAL_STRUCTURES_H


/* ************************************************************************** */
/* ************************************************************************** */
/* Section: Included Files                                                    */
/* ************************************************************************** */
/* ************************************************************************** */

#include <stddef.h>                     // Defines NULL

typedef struct __attribute__((__packed__)) data_visualiser_data
{
    uint8_t START_OF_FRAME;
    int16_t new_ecg_sample;    
    int16_t new_ecg_sample_offset_corrected;
    int16_t lp_FIR_sample;
    int16_t lp_IIR_sample;
    int16_t mV_Conversion;
    int16_t differentiated_sample;
    uint32_t squared_sample;
    uint32_t integrated_sample;    
    uint32_t pt_threshold_signal;
    uint8_t bpm_state;
    int16_t R2R_interval;
    int16_t bpm;
    int16_t bpm_average;
    int16_t bpm_min;
    int16_t bpm_max;
    int16_t qrs_width;
    int16_t p_wave_data;
    int16_t p_wave_threshold;
    uint8_t p_wave_crossings;
    uint8_t p_wave_abnormal;
    uint8_t p_wave_ok;
    uint8_t END_OF_FRAME;
}data_visualiser_data_t;


#endif /* GLOBAL_STRUCTURES_H */

/* *****************************************************************************
 End of File
 */
