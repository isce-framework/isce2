#include "image_sio.h"
#include "lib_functions.h"

void
init_from_PRM(struct PRM inputPRM, struct PRM *prm)
{
	strcpy(prm->input_file,inputPRM.input_file);
	prm->near_range = inputPRM.near_range;
	prm->RE = inputPRM.RE;
	prm->chirp_ext = inputPRM.chirp_ext;
	prm->num_patches = inputPRM.num_patches;
}
