#include "hls_vadd.hpp"

void hls_vadd (
    hls::stream<axi_s> &axi_in,
    hls::stream<axi_s> &axi_out
) {
    // A free-runing kernel; no control interfaces needed to start the operation
    #pragma HLS INTERFACE ap_ctrl_none port=return

    // Specify that the input/output signals are AXI streams (axis)
    #pragma HLS INTERFACE axis register port=axi_in name=s_axi_in
    #pragma HLS INTERFACE axis register port=axi_out name=m_axi_out

    hls::stream<data_t> stream_load;
#pragma HLS stream variable=stream_load depth=64

    hls::stream<data_t> stream_zero;
#pragma HLS stream variable=stream_zero depth=64

    hls::stream<data_t> stream_log;
#pragma HLS stream variable=stream_log depth=64

    hls::stream<data_t> stream_mod;
#pragma HLS stream variable=stream_mod depth=64

#pragma HLS dataflow

    // LoadData: Read from the AXI stream and write to the internal stream
    // Dense_NegsToZero: Read from the internal stream and write to the internal stream
    // Dense_Log: Read from the internal stream and write to the internal stream
    // Sparse_HexToIntMod: Read from the internal stream and write to the AXI streams
    // StoreData: Read from the internal stream and write to the AXI stream

    LoadData(axi_in, stream_load);

    Dense_NegsToZero(stream_load, stream_zero);

    Dense_Log(stream_zero, stream_log);
 
    // Sparse_HexToIntMod(stream_log, stream_mod); 
    Dense_Log(stream_log, stream_mod); 
 
    StoreData(stream_mod, axi_out);    
    
}
