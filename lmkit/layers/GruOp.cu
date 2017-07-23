
#section support_code
// This code is awful and should not be used. Was a test to see how fast / slow theano scan is.

__forceinline__ __device__ float sigmoid(float a) {
	return 1.0f / (1.0f + exp(-a));
}

// Cuda kernel to run gate_activation for update. All elementwise.
__global__ void gate_activation_plus_elem_fusion_inplace(unsigned int numEls,
	unsigned int nd,
	const int * dim,
	float * a_data, const int * a_str,
	const float * b_data, const int * b_str) {

	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numThreads = blockDim.x * gridDim.x;
	for (unsigned int i = idx; i < numEls; i += numThreads)
	{
		unsigned int ii = i;
		float * a_i = a_data;
		const float * b_i = b_data;
		for (unsigned int _d = 0; _d < nd; ++_d)
		{
			unsigned int d = nd - _d - 1;
			int i_d = ii % dim[d]; /* i_d is our position in the d'th dimension   */
			ii = ii / dim[d];
			a_i += i_d * a_str[d]; /* increment our a and b pointers by i_d elements */
			b_i += i_d * b_str[d];
		}
		a_i[0] = sigmoid(a_i[0] + b_i[0]);
	}
}

// Cuda kernel to run gate_activation for update. All elementwise.
__global__ void state_times_gate_activation_plus_elem_fusion_inplace(unsigned int numEls,
	unsigned int nd,
	const int * dim,
	float * a_data, const int * a_str,
	const float * b_data, const int * b_str,
	const float * c_data, const int * c_str) {

	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numThreads = blockDim.x * gridDim.x;
	for (unsigned int i = idx; i < numEls; i += numThreads)
	{
		unsigned int ii = i;
		float * a_i = a_data;
		const float * b_i = b_data;
		const float * c_i = c_data;
		for (unsigned int _d = 0; _d < nd; ++_d)
		{
			unsigned int d = nd - _d - 1;
			int i_d = ii % dim[d]; /* i_d is our position in the d'th dimension   */
			ii = ii / dim[d];
			a_i += i_d * a_str[d]; /* increment our a and b pointers by i_d elements */
			b_i += i_d * b_str[d];
			c_i += i_d * c_str[d];
		}
		a_i[0] = c_i[0] * sigmoid(a_i[0] + b_i[0]);
	}
}

// Cuda kernel to run gate_activation for update. All elementwise.
__global__ void combine_elem_fusion_inplace(unsigned int numEls,
	unsigned int nd,
	const int * dim,
	float * state_data, const int * state_str,
	const float * A_data, const int * A_str,
	const float * B_data, const int * B_str,
	const float * inp_state_slice_data, const int * inp_state_slice_str) {

	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numThreads = blockDim.x * gridDim.x;
	for (unsigned int i = idx; i < numEls; i += numThreads)
	{
		unsigned int ii = i;
		float * state_i = state_data;
		const float * A_i = A_data;
		const float * B_i = B_data;
		const float * inp_state_slice_i = inp_state_slice_data;
		for (unsigned int _d = 0; _d < nd; ++_d)
		{
			unsigned int d = nd - _d - 1;
			int i_d = ii % dim[d]; /* i_d is our position in the d'th dimension   */
			ii = ii / dim[d];
			state_i += i_d * state_str[d]; /* increment our a and b pointers by i_d elements */
			A_i += i_d * A_str[d];
			B_i += i_d * B_str[d];
			inp_state_slice_i += i_d * inp_state_slice_str[d];
		}
		state_i[0] = tanhf(B_i[0] + inp_state_slice_i[0]) * A_i[0] + state_i[0] * (1 - A_i[0]);
	}
}



#section support_code_apply

int APPLY_SPECIFIC(gated_unit_main)(
	CudaNdarray* initial_state,
	CudaNdarray* inp_state,
	CudaNdarray* inp_update,
	CudaNdarray* inp_reset,

	CudaNdarray* state_to_state,
	CudaNdarray* state_to_update,
	CudaNdarray* state_to_reset,

	CudaNdarray** output)
{
	// XXX this is horribly unsafe.
	// There NEEDS to be checking for all allocation steps
	// This WILL segfault, if the input even smells wrong.

	npy_intp dims[2];
	const int* inp_state_dims = CudaNdarray_HOST_DIMS(inp_state);
	dims[0] = inp_state_dims[1]; // Batch size
	dims[1] = inp_state_dims[2]; // Hidden state size


								 // Make some temporary storage
								 // Following the formula:
								 // reset_values = gate_activation(states * state_to_reset + reset_inputs)
								 // update_values = gate_activation(state * state_to_update + update_inputs)
								 // state_tilde = activation((states elem* reset_values) * state_to_state + input)
								 // next_state = state_tilde elem* update_values + states elem* (1 - update_values)


								 // update_dot = states * state_to_update -- <batch, n_h> << Gemm
								 //    A
								 // update_values = gate_activation(update_dot + update_inputs) -- <batch, n_h> << Elem(2), f(inp0 + inp1)
								 //    A << possible

								 // reset_dot = states * state_to_reset -- <batch, n_h> << Gemm
								 //    B
								 // reset_state = states <elem*> f(reset_dot + reset_inputs) -- <batch, n_h> <<  Elem(3) inp0 * f(inpt2 + inp3)
								 //    B
								 // pre_activation_state  = reset_state * state_to_state << Gemm
								 //    B
								 // half_next_state = activation(pre_activation_state + input) *elem update_values
								 //                   + states elem* (1-update_values) << Elem(4), f(inp0 + inp1) * inp2 + inp3 * (1 - inp2)
								 //    A or just output
								 // Release B

								 // This has 3 Gemm calls and 3 other kernel calls.

	int seq_len = inp_state_dims[0];
	// Allocate temporarys, Error check needed

	if (*output) Py_DECREF(*output);
	*output = (CudaNdarray *)CudaNdarray_New(); //Err check me?
	CudaNdarray_alloc_contiguous(*output, 2, dims); // ERR check me

	CudaNdarray *state = (CudaNdarray *)CudaNdarray_New(); //Err check me?
	CudaNdarray_alloc_contiguous(state, 2, dims); // ERR check me

												  //CudaNdarray* state = *output;
	CudaNdarray_CopyFromCudaNdarray(state, initial_state);

	CudaNdarray* A = (CudaNdarray *)CudaNdarray_New();
	CudaNdarray_alloc_contiguous(A, 2, dims); // ERR check me
	CudaNdarray* B = (CudaNdarray *)CudaNdarray_New();
	CudaNdarray_alloc_contiguous(B, 2, dims); // ERR check me

											  //TODO set intiial state somehow


	unsigned int size = dims[0] * dims[1];
	unsigned int threads_per_block = std::min(size, (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);
	unsigned int n_blocks = std::min(ceil_intdiv(size, threads_per_block), (unsigned int)NUM_VECTOR_OP_BLOCKS);

#define DOT(o, a, b) CudaNdarray_gemm(1.0f, a, b, 0.0f, o);
	// This loop is done on CPU. It could probably be moved to GPU. Torch7 is not doing this though (i think).
	for (int on_step = 0; on_step < seq_len; ++on_step) {
		// The following is random access probably. Should be a way to get better cache performance
		// with iterator style interface??
		CudaNdarray* inp_state_slice =
			(CudaNdarray*)CudaNdarray_Subscript((PyObject*)inp_state, PyInt_FromLong(on_step));
		CudaNdarray* inp_update_slice =
			(CudaNdarray*)CudaNdarray_Subscript((PyObject*)inp_update, PyInt_FromLong(on_step));
		CudaNdarray* inp_reset_slice =
			(CudaNdarray*)CudaNdarray_Subscript((PyObject*)inp_reset, PyInt_FromLong(on_step));

		DOT(A, state, state_to_update);
		gate_activation_plus_elem_fusion_inplace << <n_blocks, threads_per_block >> >(size, 2,
			CudaNdarray_DEV_DIMS(A),
			CudaNdarray_DEV_DATA(A), CudaNdarray_DEV_STRIDES(A),
			CudaNdarray_DEV_DATA(inp_update_slice), CudaNdarray_DEV_STRIDES(inp_update_slice));

		DOT(B, state, state_to_reset);

		state_times_gate_activation_plus_elem_fusion_inplace << <n_blocks, threads_per_block >> >(size, 2,
			CudaNdarray_DEV_DIMS(B),
			CudaNdarray_DEV_DATA(B), CudaNdarray_DEV_STRIDES(B),
			CudaNdarray_DEV_DATA(inp_reset_slice), CudaNdarray_DEV_STRIDES(inp_reset_slice),
			CudaNdarray_DEV_DATA(state), CudaNdarray_DEV_STRIDES(state));

		DOT(B, B, state_to_state);
		combine_elem_fusion_inplace << <n_blocks, threads_per_block >> >(size, 2,
			CudaNdarray_DEV_DIMS(state),
			CudaNdarray_DEV_DATA(state), CudaNdarray_DEV_STRIDES(state),
			CudaNdarray_DEV_DATA(A), CudaNdarray_DEV_STRIDES(A),
			CudaNdarray_DEV_DATA(B), CudaNdarray_DEV_STRIDES(B),
			CudaNdarray_DEV_DATA(inp_state_slice), CudaNdarray_DEV_STRIDES(inp_state_slice));
	}
	//CudaNdarray_CopyFromCudaNdarray(*output, state);
	*output = state;

#undef DOT
	return 0;
}