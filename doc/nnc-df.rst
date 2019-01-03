NNC Dataframe
=============

A simple dataframe API is provided to help manage data that needs to feed into the NNC graphs. Beyond that, a dataframe abstraction would work better for any machine learning framework because it handles data filtering, alignments, and batching.

Operations on the Data
----------------------

Data loaded into NNC's dataframe API as rows and columns. We can iterate through rows with selected columns. Data can be shuffled. A new column can be derived from selected other columns, per the map function you specified. Multiple rows can be reduced into one row, per reduce function you specified.

Iteration
---------

A iterator can be created with :cpp:func:`ccv_cnnp_dataframe_iter_new`. Which record to access can be controlled by :cpp:func:`ccv_cnnp_dataframe_iter_set_cursor`. Reset the cursor will invalidate prefetched records. Records can be prefetched. If a record is prefetched, :cpp:func:`ccv_cnnp_dataframe_iter_next` will simply return the record from prefetch cache.

A stream context is used to coordinate data prefetch and iteration. If the stream context is provided, both prefetch and iteration are asynchronous and happens on the given stream context. A typical use case is to get record on one stream context, pass that record to a graph for computation with this stream context, and start prefetch immediately on a different stream context. When the graph computation is done, switch to the prefetch stream context to get the record and initiate another round of graph computation. Effectively, this allows us to overlap data preparation with actual computation.

Map
---

The data preparation pipeline can be long. Thus, dataframe API prefers **pull** for such processing. When a column is requested, it pulls in the data from other columns and then performs necessary transformations. It can include move data from CPU to a GPU device, applying data augmentation etc. The map function provided does the transformation.

Reduce
------

Multiple rows can be combined with a reduce function. Because reduce function generates data with different row count, a new dataframe is created (when map, only a new column within existing dataframe added). Similarly to map, when doing reduce, the data preparation pipeline performs a pull to the data. This is typically used to batch multiple records into one tensor to feed the graph.

Others
------

Comparing with other dataframe APIs, the implementation at this time is pretty raw. We don't support sorting, or a more generic filtering (you can use reduce to do some filtering, but it is limited), or joining multiple dataframes. It is unclear to me how to implement these efficiently without worrying at certain point it becomes bottleneck. On the other hand, the above implementation is as efficient as you can get (because it is raw).

Use Dataframe with Addons
-------------------------

The raw dataframe API (i.e. :cpp:func:`ccv_cnnp_dataframe_new`, :cpp:func:`ccv_cnnp_dataframe_map` and :cpp:func:`ccv_cnnp_dataframe_reduce_new`) are hard to use. Unlike helper functions such as :cpp:func:`CPU_TENSOR_NCHW` in ``ccv_nnc_easy.h``, there is no helper functions to fill in the :cpp:class:`ccv_cnnp_column_data_t`. This is intentional. You should interact dataframe with its addons API instead.

The addons API are implemented with the raw dataframe API. It provides the ability to load data from :cpp:class:`ccv_array_t` (in the future, from CSV file and SQLite tables as well). Being able to iterate through a :cpp:class:`ccv_array_t` doesn't do much (you can do the same by iterating with for loop). The power comes from combining multiple operations on top of that. For example, if you have a :cpp:class:`ccv_array_t` of file names, you can use :cpp:func:`ccv_cnnp_dataframe_read_image` to load these files as images. Certain image jitter can be applied with :cpp:func:`ccv_cnnp_dataframe_image_random_jitter`. Finally, you can batch these images into tensors with :cpp:func:`ccv_cnnp_dataframe_batching_new`.

When you iterate through the newly created dataframe, you can get the batched tensor one by one. More importantly, using dataframe also supports stream contexts. When you prefetch on one stream context, it is done asynchronously on that stream context. Rather than using a IO thread for data loader, this is NNC's way of supporting asynchronous data loading v.s. training.

Fundamentally, using dataframe with addons enables you to express computations independent of the actual execution. The actual execution (such as reading from disk, applying image random jitter or copying to GPU) only happens when you prefetch or iterate through it. This enables the powerful abstraction to load, manipulate and feed data into NNC's training process.
