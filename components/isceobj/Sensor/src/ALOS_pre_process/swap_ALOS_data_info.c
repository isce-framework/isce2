#include "image_sio.h"
#include "lib_functions.h"
/* this swaps bytes */
#define SWAP_2(x) ( (((x) & 0xff) << 8) | ((unsigned short)(x) >> 8) )
#define SWAP_4(x) ( ((x) << 24) | \
                (((x) << 8) & 0x00ff0000) | \
                (((x) >> 8) & 0x0000ff00) | \
                ((x) >> 24) )
#define FIX_SHORT(x) (*(unsigned short *)&(x) = SWAP_2(*(unsigned short *)&(x)))
#define FIX_INT(x)   (*(unsigned int *)&(x)   = SWAP_4(*(unsigned int *)&(x)))
#define FIX_FLOAT(x) FIX_INT(x)
/*------------------------------------------------------------------*/
/* need to swap bytes for all      */
/* must be a better way to do this */
void swap_ALOS_data_info(struct sardata_info *sdr)
{
	FIX_SHORT(sdr->channel_indicator);
	FIX_SHORT(sdr->channel_code);
	FIX_SHORT(sdr->transmit_polarization);
	FIX_SHORT(sdr->receive_polarization);
	FIX_SHORT(sdr->onboard_range_compress);
	FIX_SHORT(sdr->chirp_type);
	FIX_SHORT(sdr->nought_line_flag);
	FIX_SHORT(sdr->platform_update_flag);

	FIX_INT(sdr->sequence_number);
	FIX_INT(sdr->record_length);
	FIX_INT(sdr->data_line_number);
	FIX_INT(sdr->data_record_index);
	FIX_INT(sdr->n_left_fill_pixels);
	FIX_INT(sdr->n_data_pixels);
	FIX_INT(sdr->n_right_fill_pixels);
	FIX_INT(sdr->sensor_update_flag);
	FIX_INT(sdr->sensor_acquisition_year);
	FIX_INT(sdr->sensor_acquisition_DOY);
	FIX_INT(sdr->sensor_acquisition_msecs_day);
	FIX_INT(sdr->PRF);
	FIX_INT(sdr->scan_ID);
	FIX_INT(sdr->chirp_length);
	FIX_INT(sdr->chirp_constant_coeff);
	FIX_INT(sdr->chirp_linear_coeff);
	FIX_INT(sdr->chirp_quad_coeff);
	FIX_INT(sdr->receiver_gain);
	FIX_INT(sdr->elec_antenna_squint_angle);
	FIX_INT(sdr->mech_antenna_squint_angle);
	FIX_INT(sdr->slant_range);
	FIX_INT(sdr->data_record_window_position);
	FIX_INT(sdr->platform_latitude);
	FIX_INT(sdr->platform_longitude);
	FIX_INT(sdr->platform_altitude);
	FIX_INT(sdr->platform_ground_speed);
	FIX_INT(sdr->platform_velocity_x);
	FIX_INT(sdr->platform_velocity_y);
	FIX_INT(sdr->platform_velocity_z);
	FIX_INT(sdr->platform_acc_x);
	FIX_INT(sdr->platform_acc_y);
	FIX_INT(sdr->platform_acc_z);
	FIX_INT(sdr->platform_track_angle_1);
	FIX_INT(sdr->platform_track_angle_2);
	FIX_INT(sdr->platform_pitch_angle);
	FIX_INT(sdr->platform_roll_angle);
	FIX_INT(sdr->platform_yaw_angle);
	if (ALOS_format == 0) FIX_INT(sdr->frame_counter);
}
/*------------------------------------------------------------------*/
