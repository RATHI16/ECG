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

#ifndef QRS_ALGORITHM_H    /* Guard against multiple inclusion */
#define QRS_ALGORITHM_H

#include <stddef.h>                     // Defines NULL
#include "DV_structure.h"



/* ************************************************************************** */
/* ************************************************************************** */
/* Section: Included Files                                                    */
/* ************************************************************************** */
/* ************************************************************************** */

/* This section lists the other files that are included in this file.
 */

void detect_qrs(data_visualiser_data_t *data_packet);
int16_t get_bpm_average();
int16_t get_bpm_min();
int16_t get_bpm_max();

void insert_data_to_p_wave_buffer(int16_t input_sample); 
void read_p_wave_buffer_to_output(data_visualiser_data_t *data_packet);

#endif /* QRS_ALGORITHM_H */

/* *****************************************************************************
 End of File
 */
