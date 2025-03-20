#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include <stdint.h>
#include <ap_int.h>
#include <hls_math.h>

// Constants and typedefs
#define AXI_DATA_BITS 512
typedef ap_axiu<AXI_DATA_BITS, 0, 0, 0> axi_s;

#define FLOAT_BITS 32
#define NUM_FLOATS AXI_DATA_BITS / FLOAT_BITS

typedef union
{
    float float32;
    uint32_t uint32;
} conv;

/**
 * hls_vadd example
 * @brief Reads floats from the two incoming streams and addes them, storing the result to axi_out
 * 
 * @param[in] axi_in Incoming AXI stream, corresponding to vector
 * @param[out] axi_out Outgoing AXI stream; resulting vector
 *
 * The inputs/outputs are actually buffered in a hls::stream, which corresponds to a FIFO
 * Each element in the FIFO is a AXI stream with data, valid etc. signals
 */
void hls_vadd (
    hls::stream<axi_s> &axi_in,
    hls::stream<axi_s> &axi_out
);

void LoadData(
    hls::stream<axi_s> &axi_in, 
    hls::stream<ap_uint<512> >& stream_out 
) {
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

        axi_s x;
        ap_uint<512> y;

        if(!axi_in.empty()) {
            x = axi_in.read();
            y = x.data;
            stream_out.write(y);
        }

}

void Dense_NegsToZero(
    hls::stream<ap_uint<512> >& stream_in,
    hls::stream<ap_uint<512> >& stream_out 
) {
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

        ap_uint<512> input_data;
        ap_uint<512> output_data;

        Dense_NegsToZero:
        if(!stream_in.empty()) {

            input_data = stream_in.read();

            for (int k = 0; k < 16; k++) {
                #pragma HLS unroll

                int tmp_value;
                conv conv_value;

                tmp_value = input_data(32*k+31, 32*k);
                tmp_value = (tmp_value < 0) ? 0 : tmp_value;
                output_data(32*k+31, 32*k) = tmp_value;

            }

			stream_out.write(output_data);
		}

}

void Dense_Log(
    hls::stream<ap_uint<512> >& stream_in,
    hls::stream<ap_uint<512> >& stream_out 
) {
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off


    ap_uint<512> input_data = 0;
    ap_uint<512> output_data = 0;

    Dense_Log:
    if(!stream_in.empty()) {

        input_data = stream_in.read();

        for (int j = 0; j < 16; j++) {
            #pragma HLS unroll

            int tmp_value;
            conv conv_value;

            tmp_value = input_data(32*j+31, 32*j);
            conv_value.float32 = hls::logf(tmp_value+1);
            output_data(32*j+31, 32*j) = conv_value.uint32; 

        }

		stream_out.write(output_data);
	}

}

void Sparse_HexToIntMod(
    hls::stream<ap_uint<512> >& stream_in,
    hls::stream<ap_uint<512> >& stream_out
) {
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

    ap_uint<512> input_data = 0;
    ap_uint<512> output_data = 0;

    Sparse_HexToIntMod:
    if(!stream_in.empty()) {

        input_data = stream_in.read();

        for (int j = 0; j < 16; j++) {
            #pragma HLS unroll

            ap_uint<32> tmp_input = input_data(32*j+31, 32*j);
            ap_uint<32> tmp_output = tmp_input & 0x3FF; // % 1024
            output_data(32*j+31, 32*j) = tmp_output; 

        }

		stream_out.write(output_data);
	}
}

void StoreData(
    hls::stream<ap_uint<512> >& stream_in,
    hls::stream<axi_s> &axi_out
) {
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

    // ap_uint<512> input_data = 0;
    axi_s y;

    StoreData:
    if(!stream_in.empty()) {

		y.data = stream_in.read();
		y.keep = 0xF;
		y.id = 0;
		y.last = 0;
		axi_out.write(y);
	}
}