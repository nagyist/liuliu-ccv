/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_cnnp_model_internal_h
#define GUARD_ccv_cnnp_model_internal_h

#include "ccv_nnc.h"
#include "_ccv_nnc_stream.h"
#include "_ccv_nnc_xpu_alloc.h"
#include "3rdparty/khash/khash.h"

typedef void(*ccv_cnnp_cmd_updater_f)(void* const context, const ccv_nnc_graph_exec_symbol_t symbol, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint);
typedef void(*ccv_cnnp_add_to_array_f)(void* const context, const ccv_nnc_tensor_symbol_t symbol, const int is_trainable);
/**
 * This is the virtual table of the model.
 */
typedef struct {
	void (*deinit)(ccv_cnnp_model_t* const self); /**< It can be nil. */
	void (*build)(ccv_cnnp_model_t* const self, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size); /**< Call this graph to build computation. No need to specify input size or output size, as it is defined along in the model already. */
	void (*init_states)(ccv_cnnp_model_t* const self, ccv_nnc_symbolic_graph_t* const graph, const ccv_cnnp_state_initializer_f initializer, void* const context); /**< This is called to init ccv_nnc_tensor_symbol_t with a exec. */
	void (*add_to_parameter)(ccv_cnnp_model_t* const self, const ccv_cnnp_add_to_array_f add_to_array, void* const parameters, const int is_trainable); /**< This is called to add ccv_nnc_tensor_symbol_t to as list of parameters. */
	void (*add_to_output)(ccv_cnnp_model_t* const self, const ccv_cnnp_add_to_array_f add_to_array, void* const outputs); /**< This is called to add ccv_nnc_tensor_symbol_t to as list of outputs for retention. The final outputs are already added. This method is optional for any additional values we want to retain. */
	ccv_cnnp_model_t* (*copy)(const ccv_cnnp_model_t* const self, void* const context); /**< This is called to make a deep copy of itself. */
	void (*set_is_test)(ccv_cnnp_model_t* const self, const int is_test, const ccv_cnnp_cmd_updater_f updater, void* const context); /**< This is called when it is switched between test or training. */
	void (*add_to_parameter_indices)(ccv_cnnp_model_t* const self, const int index, ccv_array_t* const parameter_indices); /**< This is called when we try to get parameter indices out of a given model */
	void (*notify)(const ccv_cnnp_model_t* const self, const int tag, void* const payload); /**< This is called when we want to notify something to this model. */
} ccv_cnnp_model_vtab_t;

struct ccv_cnnp_model_io_s {
	int param_ref; // Reference to parameter in the model, starts with 1. 0 means no such thing.
	int param_sel; // Selector to parameter in the model, starts with 1. 0 means no selector.
	int visit; // Temporary bits stored in the ccv_cnnp_model_io_t object, whoever uses it should clean it up.
	ccv_cnnp_model_t* model; // Reference back to the model who holds it. This is required because the model is the one whole holds the io.
	ccv_array_t* incomings; // Array of ccv_cnnp_model_io_t. The order is important because it impacts the order of symbols.
	ccv_array_t* dependencies; // Array of ccv_cnnp_model_io_t.
	int dependents; // Number of dependents.
	ccv_array_t* outgoings; // Array of ccv_cnnp_model_io_t.
	ccv_nnc_tensor_symbol_t* outputs; // This is different from the outputs from a model. A model could be reused, causing the outputs on that model to be the most recent one. This keeps the outputs of each.
};

enum {
	CCV_CNNP_MODEL_GRAPH_FIT_MODE, // This mode computes loss, backprop, and then apply gradients.
	CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE_NO_GRAD, // This mode allows you to only use ccv_cnnp_model_evaluate (others require gradient).
	CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE, // This mode allows you to use ccv_cnnp_model_evaluate, ccv_cnnp_model_backward, ccv_cnnp_model_apply_gradients separately.
};

enum {
	CCV_CNNP_COMPILED_DATA_GRADIENT_NONE,
	CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES,
	CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES_AND_INPUTS,
};

enum {
	CCV_CNNP_REWIND_GRAPH_EXEC,
	CCV_CNNP_REWIND_TENSOR,
};

typedef struct {
	int type;
	union {
		ccv_nnc_tensor_symbol_t tensor;
		ccv_nnc_graph_exec_symbol_t graph_exec;
	};
} ccv_cnnp_rewind_symbol_t;

#define CCV_NNC_TENSOR(tv) ((ccv_nnc_tensor_t*)((uintptr_t)(tv) & ~(uintptr_t)1))
#define CCV_NNC_INIT_V(v) ((uint32_t*)((uintptr_t)(v) & ~(uintptr_t)1))

// This contains relevant information after model compilation.
typedef struct {
	int graph_mode;
	int gradient_mode; // Have init gradient graph.
	int is_test;
	int stream_type;
	int outgrad_size;
	uint64_t disable_outgrad;
	ccv_nnc_symbolic_graph_compile_param_t compile_params;
	ccv_nnc_xpu_alloc_t xpu_alloc;
	ccv_nnc_graph_t* graph;
	ccv_nnc_tensor_arena_t* tensor_arena;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena;
	khash_t(stream_map)* stream_map; // Keeps track of streams on both GPU / CPU and devices so it can be used properly during execution.
	ccv_array_t* parameters;
	uint64_t* parameter_flags;
	ccv_array_t* internals; // Additional symbols need to retain.
	ccv_nnc_tensor_symbol_t* gradients;
	ccv_nnc_tensor_symbol_t* outgrads;
	ccv_nnc_tensor_symbol_t* updated_parameters;
	ccv_nnc_graph_exec_symbol_t* update_nodes;
	ccv_nnc_tensor_symbol_map_t* saved_aux;
	ccv_array_t* rewindables;
	ccv_array_t* gradient_checkpoints;
	struct {
		int size;
		uint32_t* v; // If the last is 1, we know it is incomplete (thus, the tensors_init_1 hasn't been called yet. This is to save RAM usage.
	} tensors_init;
	struct {
		ccv_nnc_tensor_t** internals; // Additional need to retained tensors.
		ccv_nnc_tensor_t** parameters;
		ccv_nnc_tensor_t** gradients;
		ccv_nnc_tensor_t** accum_gradients;
	} tensors;
	struct {
		ccv_array_t* parameters;
		ccv_array_t* internals;
	} ids;
	struct {
		int to_op_size;
		int to_size;
		ccv_nnc_graph_exec_t* to_ops;
		ccv_nnc_graph_exec_symbol_t* tos;
		ccv_nnc_graph_static_schedule_t* schedule; // The partial schedule for running evaluate step.
	} evaluate; // Data related to ccv_cnnp_model_evaluate
	struct {
		int count; // Called backward how many times. Starting with 0.
		int from_op_size;
		ccv_nnc_graph_exec_t* from_ops; // These are the ops in the main graph.
		int to_size;
		ccv_nnc_graph_exec_symbol_t* tos;
		ccv_nnc_graph_t* accum; // The graph to accumulate gradients.
		ccv_nnc_tensor_arena_t* tensor_arena;
		ccv_nnc_graph_exec_arena_t* graph_exec_arena;
		ccv_nnc_tensor_symbol_t* gradients; // The new gradients.
		ccv_nnc_tensor_symbol_t* accum_gradients; // The old accumulate gradients.
		ccv_nnc_tensor_symbol_t* updated_accum_gradients; // The new accumulate gradients.
		ccv_nnc_graph_static_schedule_t* schedule; // The partial schedule for running backward step.
	} backward;
	struct {
		ccv_nnc_graph_t* graph;
		ccv_nnc_tensor_arena_t* tensor_arena;
		ccv_nnc_graph_exec_arena_t* graph_exec_arena;
	} apply_gradients;
	struct {
		ccv_nnc_cmd_t minimizer;
		ccv_array_t* parameters;
		int max_saved_aux_size;
	} minimize;
	ccv_nnc_cmd_t loss;
	ccv_nnc_tensor_symbol_t* f;
	ccv_nnc_tensor_symbol_t fits[1];
} ccv_cnnp_compiled_data_t;

struct ccv_cnnp_model_s {
	const ccv_cnnp_model_vtab_t* isa;
	int input_size; // This is the best effort number, mostly just for subclass to use.
	int output_size;
	int max_stream_count;
	ccv_array_t* io; // The opaque io that can be nil.
	ccv_array_t* parameter_indices; // The indexes for parameters in the final model.
	ccv_nnc_symbolic_graph_t* graph;
	ccv_nnc_tensor_symbol_t* inputs; // Unlike outputs, which is not dynamically allocated, inputs is dynamically allocated, and may be 0.
	ccv_nnc_tensor_symbol_t* outputs;
	char* name;
	struct {
		ccv_cnnp_model_notify_f func;
		void* context;
	} notify_hook;
	ccv_cnnp_compiled_data_t* compiled_data;
	int parallel_count; // How many parallel devices.
	int memory_compression; // Whether to enable memory compression for training phase.
	int gradient_checkpointing; // Whether to enable gradient checkpointing for training phase.
	int is_trainable; // Whether this model can be trained or not.
	int memory_reduction; // Whether to enable memory reduction techniques for training phase.
	int exec_flags; // The flags to be applied to the execution nodes.
	size_t workspace_size; // Set the default workspace size.
	struct {
		ccv_cnnp_model_io_reader_f reader;
		ccv_cnnp_model_io_writer_f writer;
	} rw;
	void* data; // Temporary storage for some internal functions.
};

KHASH_MAP_INIT_STR(ccv_cnnp_model_name_bank, int)

typedef struct {
	int sequence;
	khash_t(ccv_cnnp_model_name_bank)* bank;
	const char* name;
} ccv_cnnp_model_name_t;

typedef struct {
	int it;
	ccv_cnnp_model_t* model;
	khash_t(ccv_cnnp_model_name_bank)* bank;
	ccv_array_t* sequences;
} ccv_cnnp_model_sequence_t;

static inline void ccv_cnnp_model_push(ccv_cnnp_model_t* const self, ccv_cnnp_model_sequence_t* const model_sequence)
{
	// Reset to 0.
	if (!model_sequence->sequences)
		model_sequence->sequences = ccv_array_new(sizeof(ccv_cnnp_model_name_t), 1, 0);
	khash_t(ccv_cnnp_model_name_bank)* bank = model_sequence->sequences->rnum > 0 ? ((ccv_cnnp_model_name_t*)ccv_array_get(model_sequence->sequences, model_sequence->sequences->rnum - 1))->bank : model_sequence->bank;
	int ret;
	khiter_t k = kh_put(ccv_cnnp_model_name_bank, bank, self->name ? self->name : "", &ret);
	int sequence;
	if (ret != 0)
		sequence = kh_val(bank, k) = 0;
	else
		sequence = ++kh_val(bank, k);
	ccv_cnnp_model_name_t name = {
		.bank = kh_init(ccv_cnnp_model_name_bank),
		.name = self->name,
		.sequence = sequence,
	};
	ccv_array_push(model_sequence->sequences, &name);
	model_sequence->model = self;
}

static inline void ccv_cnnp_model_pop(const ccv_cnnp_model_t* const self, ccv_cnnp_model_sequence_t* const model_sequence)
{
	khash_t(ccv_cnnp_model_name_bank)* const bank = ((ccv_cnnp_model_name_t*)ccv_array_get(model_sequence->sequences, model_sequence->sequences->rnum - 1))->bank;
	kh_destroy(ccv_cnnp_model_name_bank, bank);
	--model_sequence->sequences->rnum;
	assert(model_sequence->sequences->rnum >= 0);
	model_sequence->model = 0;
}

static inline ccv_cnnp_model_t* _ccv_cnnp_model_copy(const ccv_cnnp_model_t* const model, void* const context)
{
	assert(model->isa->copy);
	ccv_cnnp_model_t* const copy = model->isa->copy(model, context);
	copy->parallel_count = model->parallel_count;
	copy->memory_compression = model->memory_compression;
	copy->memory_reduction = model->memory_reduction;
	copy->max_stream_count = model->max_stream_count;
	copy->gradient_checkpointing = model->gradient_checkpointing;
	return copy;
}

static inline void ccv_cnnp_model_copy_name(ccv_cnnp_model_t* const self, const char* const name)
{
	if (name)
	{
		const size_t len = strnlen(name, 63);
		const size_t n = len + 1;
		self->name = (char*)ccmalloc(n);
		// Don't use strndup because this way I can have custom allocator (for ccmalloc).
		memcpy(self->name, name, n);
		self->name[len] = 0;
	}
}

static inline void ccv_cnnp_model_add_to_parameter(ccv_cnnp_model_t* const self, const ccv_cnnp_add_to_array_f add_to_array, void* const parameters, const int is_trainable)
{
	if (self->isa->add_to_parameter)
		self->isa->add_to_parameter(self, add_to_array, parameters, is_trainable);
}

static inline void ccv_cnnp_model_add_to_output(ccv_cnnp_model_t* const self, const ccv_cnnp_add_to_array_f add_to_array, void* const outputs)
{
	if (self->isa->add_to_output)
		self->isa->add_to_output(self, add_to_array, outputs);
}

typedef struct {
	int exec_flags;
	int is_trainable;
	int is_gradient_checkpointing;
	ccv_cnnp_model_sequence_t* model_sequence;
	ccv_cnnp_add_to_array_f add_to_array;
	ccv_array_t* parameters;
	struct {
		void* add_to_parameter;
		void* add_to_output;
	} context;
	ccv_array_t* gradient_checkpoints;
} ccv_cnnp_model_build_data_t; // Host temporary data for building models.

typedef struct {
	int input_size;
	int output_size;
	int is_trainable;
	ccv_cnnp_model_t* model;
	void (*build)(ccv_cnnp_model_t* const self, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size);
	ccv_array_t* tensor_symbols;
	ccv_nnc_tensor_symbol_t* inputs;
	ccv_nnc_tensor_symbol_t* outputs;
} ccv_cnnp_model_gradient_checkpoint_t;

static inline ccv_nnc_tensor_symbol_t ccv_cnnp_parameter_from_indice(ccv_cnnp_model_t* const self, const int indice)
{
	assert(self->data);
	ccv_cnnp_model_build_data_t* const build_data = (ccv_cnnp_model_build_data_t*)self->data;
	assert(indice < build_data->parameters->rnum);
	return *(ccv_nnc_tensor_symbol_t*)ccv_array_get(build_data->parameters, indice);
}

typedef struct {
	int record;
	ccv_array_t* tensor_symbols;
	void* old_tensor_symbol_new_hook_context;
	ccv_nnc_tensor_symbol_new_hook_f old_tensor_symbol_new_hook;
	void* old_tensor_symbol_alias_new_hook_context;
	ccv_nnc_tensor_symbol_alias_new_hook_f old_tensor_symbol_alias_new_hook;
} ccv_cnnp_model_gradient_checkpoint_build_context_t;

static void _ccv_cnnp_model_gradient_checkpoint_tensor_symbol_new_hook(void* context, const ccv_nnc_tensor_symbol_t symbol, const ccv_nnc_tensor_param_t info, const char* const name)
{
	ccv_cnnp_model_gradient_checkpoint_build_context_t* const build_context = (ccv_cnnp_model_gradient_checkpoint_build_context_t*)context;
	if (build_context->record)
		ccv_array_push(build_context->tensor_symbols, &symbol);
	if (build_context->old_tensor_symbol_new_hook)
		build_context->old_tensor_symbol_new_hook(build_context->old_tensor_symbol_new_hook_context, symbol, info, name);
}

static void _ccv_cnnp_model_gradient_checkpoint_tensor_symbol_alias_new_hook(void* context, const ccv_nnc_tensor_symbol_t symbol, const ccv_nnc_tensor_symbol_t from_symbol, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int inc[CCV_NNC_MAX_DIM_ALLOC], const ccv_nnc_tensor_param_t info, const char* const name)
{
	ccv_cnnp_model_gradient_checkpoint_build_context_t* const build_context = (ccv_cnnp_model_gradient_checkpoint_build_context_t*)context;
	if (build_context->record)
		ccv_array_push(build_context->tensor_symbols, &symbol);
	if (build_context->old_tensor_symbol_alias_new_hook)
		build_context->old_tensor_symbol_alias_new_hook(build_context->old_tensor_symbol_alias_new_hook_context, symbol, from_symbol, ofs, inc, info, name);
}

static inline void ccv_cnnp_model_build(ccv_cnnp_model_t* const self, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	assert(self->data);
	ccv_cnnp_model_build_data_t* const build_data = (ccv_cnnp_model_build_data_t*)self->data;
	const int old_exec_flags = build_data->exec_flags;
	const int old_is_trainable = build_data->is_trainable;
	if (self->exec_flags)
		build_data->exec_flags |= self->exec_flags;
	if (self->is_trainable >= 0)
		build_data->is_trainable = self->is_trainable;
	if (self->name && self->name[0] != '\0')
		ccv_cnnp_model_push(self, build_data->model_sequence);
	if (self->gradient_checkpointing == 1 && !build_data->is_gradient_checkpointing)
	{
		build_data->is_gradient_checkpointing = 1;
		// Prepare to record gradient checkpoint. We will log the build function, inputs, what are the tensors / graph execs we created.
		if (!build_data->gradient_checkpoints)
			build_data->gradient_checkpoints = ccv_array_new(sizeof(ccv_cnnp_model_gradient_checkpoint_t), 0, 0);
		ccv_cnnp_model_gradient_checkpoint_build_context_t build_context = {
			.record = 1,
			.tensor_symbols = ccv_array_new(sizeof(ccv_nnc_tensor_symbol_t), 0, 0),
		};
		build_context.old_tensor_symbol_new_hook_context = ccv_nnc_tensor_symbol_new_hook(graph, _ccv_cnnp_model_gradient_checkpoint_tensor_symbol_new_hook, &build_context, &build_context.old_tensor_symbol_new_hook);
		build_context.old_tensor_symbol_alias_new_hook_context = ccv_nnc_tensor_symbol_alias_new_hook(graph, _ccv_cnnp_model_gradient_checkpoint_tensor_symbol_alias_new_hook, &build_context, &build_context.old_tensor_symbol_alias_new_hook);
		if (outputs && output_size)
		{
			assert(output_size == self->output_size);
			self->isa->build(self, graph, inputs, input_size, outputs, output_size);
			memcpy(self->outputs, outputs, sizeof(ccv_nnc_tensor_symbol_t) * output_size);
		} else
			self->isa->build(self, graph, inputs, input_size, self->outputs, self->output_size);
		ccv_nnc_tensor_symbol_new_hook(graph, build_context.old_tensor_symbol_new_hook, build_context.old_tensor_symbol_new_hook_context, 0);
		ccv_nnc_tensor_symbol_alias_new_hook(graph, build_context.old_tensor_symbol_alias_new_hook, build_context.old_tensor_symbol_alias_new_hook_context, 0);
		ccv_cnnp_model_gradient_checkpoint_t checkpoint = {
			.input_size = input_size,
			.output_size = (outputs && output_size > 0) ? output_size : self->output_size,
			.is_trainable = build_data->is_trainable,
			.model = self,
			.build = self->isa->build,
			.tensor_symbols = build_context.tensor_symbols,
			.inputs = ccmalloc(sizeof(ccv_nnc_tensor_symbol_t) * (input_size + ((outputs && output_size > 0) ? output_size : self->output_size))),
		};
		checkpoint.outputs = checkpoint.inputs + input_size;
		if (input_size > 0)
			memcpy(checkpoint.inputs, inputs, sizeof(ccv_nnc_tensor_symbol_t) * input_size);
		if (outputs && output_size > 0)
			memcpy(checkpoint.outputs, outputs, sizeof(ccv_nnc_tensor_symbol_t) * output_size);
		else if (self->outputs && self->output_size > 0)
			memcpy(checkpoint.outputs, self->outputs, sizeof(ccv_nnc_tensor_symbol_t) * self->output_size);
		ccv_array_push(build_data->gradient_checkpoints, &checkpoint);
		build_data->is_gradient_checkpointing = 0;
	} else {
		// If we want to disable gradient checkpointing for this model, we simply not log any new tensors created here so there is no mapping for these.
		int old_record;
		ccv_cnnp_model_gradient_checkpoint_build_context_t* build_context = 0;
		if (build_data->is_gradient_checkpointing)
		{
			if (self->gradient_checkpointing == -1)
			{
				ccv_nnc_tensor_symbol_new_hook_f old_tensor_symbol_new_hook;
				build_context = ccv_nnc_tensor_symbol_new_hook(graph, 0, 0, &old_tensor_symbol_new_hook);
				// Set back the build_context.
				ccv_nnc_tensor_symbol_new_hook(graph, old_tensor_symbol_new_hook, build_context, 0);
				old_record = build_context->record;
				build_context->record = 0;
			} else if (self->gradient_checkpointing == 1) { // Force to turn on gradient checkpointing if it is inside a gradient checkpointing = -1.
				ccv_nnc_tensor_symbol_new_hook_f old_tensor_symbol_new_hook;
				build_context = ccv_nnc_tensor_symbol_new_hook(graph, 0, 0, &old_tensor_symbol_new_hook);
				// Set back the build_context.
				ccv_nnc_tensor_symbol_new_hook(graph, old_tensor_symbol_new_hook, build_context, 0);
				old_record = build_context->record;
				build_context->record = 1;
			}
		}
		// No push checkpoint, easy.
		if (outputs && output_size)
		{
			assert(output_size == self->output_size);
			self->isa->build(self, graph, inputs, input_size, outputs, output_size);
			memcpy(self->outputs, outputs, sizeof(ccv_nnc_tensor_symbol_t) * output_size);
		} else
			self->isa->build(self, graph, inputs, input_size, self->outputs, self->output_size);
		if (build_context) // Restore previous state even if our gradient checkpointing controlled whether to turn on recording or not.
			build_context->record = old_record;
	}
	// Skip if there is none. This helps to load parameters to a different model when only changes non-parameterized settings (add reshapes, permutations etc).
	// If it is named, we have to push too.
	if (self->isa->add_to_parameter || self->isa->add_to_output)
	{
		// If we already pushed, no need to push again.
		if (!(self->name && self->name[0] != '\0'))
			ccv_cnnp_model_push(self, build_data->model_sequence);
		build_data->model_sequence->it = 0;
		ccv_cnnp_model_add_to_parameter(self, build_data->add_to_array, build_data->context.add_to_parameter, build_data->is_trainable);
		build_data->model_sequence->it = 0;
		ccv_cnnp_model_add_to_output(self, build_data->add_to_array, build_data->context.add_to_output);
		ccv_cnnp_model_pop(self, build_data->model_sequence);
	} else if (self->name && self->name[0] != '\0')
		ccv_cnnp_model_pop(self, build_data->model_sequence);
	build_data->exec_flags = old_exec_flags;
	build_data->is_trainable = old_is_trainable;
}

static inline void ccv_cnnp_model_init_states(ccv_cnnp_model_t* const self, ccv_nnc_symbolic_graph_t* const graph, const ccv_cnnp_state_initializer_f initializer, void* const context)
{
	if (self->isa->init_states)
		self->isa->init_states(self, graph, initializer, context);
}

static inline void ccv_cnnp_model_set_is_test(ccv_cnnp_model_t* const self, const int is_test, const ccv_cnnp_cmd_updater_f updater, void* const context)
{
	if (self->isa->set_is_test)
		self->isa->set_is_test(self, is_test, updater, context);
}

static inline void ccv_cnnp_model_add_to_parameter_indices(ccv_cnnp_model_t* const self, const int index, ccv_array_t* const parameter_indices)
{
	if (self->isa->add_to_parameter_indices)
		self->isa->add_to_parameter_indices(self, index, parameter_indices);
	else {
		int i;
		if (!self->parameter_indices)
			return;
		if (index == -1)
			for (i = 0; i < self->parameter_indices->rnum; i++)
				ccv_array_push(parameter_indices, ccv_array_get(self->parameter_indices, i));
		else if (index < self->parameter_indices->rnum)
			ccv_array_push(parameter_indices, ccv_array_get(self->parameter_indices, index));
	}
}

typedef struct {
	uint8_t add_parameter_indices;
	char prefix;
	ccv_cnnp_model_sequence_t* sequence;
	ccv_array_t* symbols;
	ccv_array_t* ids;
	ccv_array_t* trainables;
} ccv_cnnp_model_add_to_array_context_t;

void ccv_cnnp_model_tensors_init_0(const ccv_cnnp_model_t* const model, ccv_cnnp_compiled_data_t* const compiled_data);
void ccv_cnnp_model_tensors_init_1(const ccv_cnnp_model_t* const model, ccv_cnnp_compiled_data_t* const compiled_data);
int ccv_cnnp_model_tensors_any_to_alloc(const ccv_cnnp_model_t* const model, ccv_cnnp_compiled_data_t* const compiled_data);
ccv_nnc_stream_context_t* ccv_cnnp_compiled_data_get_stream(ccv_cnnp_compiled_data_t* const compiled_data, const int type);
void ccv_cnnp_model_gradient_checkpoints_cleanup_after_build(ccv_cnnp_compiled_data_t* const compiled_data, ccv_nnc_symbolic_graph_t* const graph);
void ccv_cnnp_model_apply_gradient_checkpoints(ccv_cnnp_compiled_data_t* const compiled_data, ccv_nnc_symbolic_graph_t* const graph);
void ccv_cnnp_model_add_to_array(void* const context, const ccv_nnc_tensor_symbol_t symbol, const int is_trainable);

#endif
