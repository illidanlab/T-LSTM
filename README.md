# T-LSTM
Regularity of the duration between consecutive elements of a sequence is a property that does not always hold. An architecture that can overcome this irregularity is necessary to increase the prediction performance.

Time Aware LSTM (T-LSTM) was designed to handle irregular elapsed times. T-LSTM is proposed to incorporate the elapsed time
information into the standard LSTM architecture to be able to capture the temporal dynamics of sequential data with time irregularities. T-LSTM decomposes memory cell into short-term and long-term components, discounts the short-term memory content using a non-increasing function of the elapsed time, and then combines it with the long-term memory.  
