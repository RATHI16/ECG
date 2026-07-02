/* 
 * File:   PanTompkins_QRS.h
 * Author: M19216
 *
 * Created on May 8, 2024, 5:52 PM
 */

#ifndef PANTOMPKINS_QRS_H
#define	PANTOMPKINS_QRS_H

#include <stddef.h>                     // Defines NULL


#define MWI_WINDOW_SIZE 50 // Example window size for integration
#define MWD_WINDOW_SIZE 10 // Example window size for differentiation


// Function prototypes
int16_t offset_filter(int16_t input_sample);
int16_t lp_FIR_filter_50Hz(int16_t input_sample);
int16_t lp_FIR_filter_60Hz(int16_t input_sample);
int16_t moving_window_differentiate(int16_t input_sample);
uint32_t square_wave(int16_t input_sample);
uint32_t moving_window_integration(uint32_t input_sample);
uint32_t pt_signal_max_filter(uint32_t pt_output_sample);


#endif	/* PANTOMPKINS_QRS_H */

