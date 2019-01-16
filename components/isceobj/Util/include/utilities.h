/* Function prototypes for utility routines.
 */

extern	double	wc_second();

extern	double	wc_second_();

extern	double	us_second();

extern	double	us_second_();

extern	double	second(
			int	want_us);

extern	double	second_(
			int	*want_us);

extern	int	mhz();

extern	int	mhz_();

extern	int	loops(
			double	target,
			double	opsPerCycle,
			double	totalOps);

extern	int	loops_(
			double	*target,
			double	*opsPerCycle,
			double	*totalOps);

extern	double	pctPeak(
			double	seconds,
			double	opsPerCycle,
			double	totalOps);

extern	double	pctPeak_(
			double	*seconds,
			double	*opsPerCycle,
			double	*totalOps);

typedef	double	t_peak;
typedef	double	t_cnt;

extern	void	name_bin(
			int	bin_number,
			char	*name,
			t_peak	peak);

extern	void	start_profile(
			int	bin_number);

extern	void	end_profile(
			int	bin_number,
			t_cnt	p_op_cnt);

extern	void	dump_profile();

extern	void	name_bin_(
			int	*bin_number,
			char	*name,
			t_peak	*peak,
			int	len);

extern	void	start_profile_(
			int	*bin_number);

extern	void	end_profile_(
			int	*bin_number,
			t_cnt	*p_op_cnt);

extern	void	dump_profile_();

extern	int	p_setup_(
			int *nthp);

extern	int	p_setup(
			int nth);

extern	void	mp_super_(
			int *ithp);

extern	void	mp_super(
			int ith);

extern	void	mp_unsuper_(
			int *ithp);

extern	void	mp_unsuper(
			int ith);
