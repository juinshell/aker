unsigned short read16u(FILE *f)
{
    int n;

    n = fgetc(f);
    n += fgetc(f) << 8;

    return n;
}

short read16i(FILE *f)
{
    int n;

    n = fgetc(f);
    n += fgetc(f) << 8;

    return n;
}


struct image_i16 *load_image(char *filename)
{
    FILE *infile;
    short *data;
    int w;
    int h;

    infile = fopen(filename, "r");

    if (!infile)
    {
      fprintf(stderr, "Cannot find file '%s'\n", filename);
      exit(-1);
    }

    /* Read image dimensions */
    w = read16u(infile);
    h = read16u(infile);

    /* Read image contents */
    data = (short *)malloc(w * h * sizeof(short));
    fread(data, sizeof(short), w * h, infile);

    fclose(infile);

    /* Create the return data structure */
    {
        struct image_i16 *ret =
            (struct image_i16 *)malloc(sizeof(struct image_i16));
        ret->width = w;
        ret->height = h;
        ret->data = data;
        return ret;
    }
}


__global__ void ori_larger_sad_calc_8(unsigned short *blk_sad,
    int mb_width,
    int mb_height)
{
    int tx = threadIdx.y & 1;
    int ty = threadIdx.y >> 1;

    /* Macroblock and sub-block coordinates */
    int mb_x = blockIdx.x;
    int mb_y = blockIdx.y;

    /* Number of macroblocks in a frame */
    int macroblocks = __mul24(mb_width, mb_height);
    int macroblock_index = (__mul24(mb_y, mb_width) + mb_x) * MAX_POS_PADDED;

    int search_pos;

    unsigned short *bi;
    unsigned short *bo_6, *bo_5, *bo_4;

    bi = blk_sad
    + (__mul24(macroblocks, 25) + (ty * 8 + tx * 2)) * MAX_POS_PADDED
    + macroblock_index * 16;

    // Block type 6: 4x8
    bo_6 = blk_sad
    + ((macroblocks << 4) + macroblocks + (ty * 4 + tx * 2)) * MAX_POS_PADDED
    + macroblock_index * 8;

    if (ty < 100) // always true, but improves register allocation
    {
        // Block type 5: 8x4
        bo_5 = blk_sad
        + ((macroblocks << 3) + macroblocks + (ty * 4 + tx)) * MAX_POS_PADDED
        + macroblock_index * 8;

        // Block type 4: 8x8
        bo_4 = blk_sad
        + ((macroblocks << 2) + macroblocks + (ty * 2 + tx)) * MAX_POS_PADDED
        + macroblock_index * 4;
    }

    for (search_pos = threadIdx.x; search_pos < (MAX_POS+1)/2; search_pos += 32)
    {
        /* Each uint is actually two 2-byte integers packed together.
        * Only addition is used and there is no chance of integer overflow
        * so this can be done to reduce computation time. */
        uint i00 = ((uint *)bi)[search_pos];
        uint i01 = ((uint *)bi)[search_pos + MAX_POS_PADDED/2];
        uint i10 = ((uint *)bi)[search_pos + 4*MAX_POS_PADDED/2];
        uint i11 = ((uint *)bi)[search_pos + 5*MAX_POS_PADDED/2];

        ((uint *)bo_6)[search_pos]                  = i00 + i10;
        ((uint *)bo_6)[search_pos+MAX_POS_PADDED/2] = i01 + i11;
        ((uint *)bo_5)[search_pos]                  = i00 + i01;
        ((uint *)bo_5)[search_pos+2*MAX_POS_PADDED/2] = i10 + i11;
        ((uint *)bo_4)[search_pos]                  = (i00 + i01) + (i10 + i11);
    }
}

__global__ void ori_larger_sad_calc_16(unsigned short *blk_sad,
     int mb_width,
     int mb_height)
{
    /* Macroblock coordinates */
    int mb_x = blockIdx.x;
    int mb_y = blockIdx.y;

    /* Number of macroblocks in a frame */
    int macroblocks = __mul24(mb_width, mb_height) * MAX_POS_PADDED;
    int macroblock_index = (__mul24(mb_y, mb_width) + mb_x) * MAX_POS_PADDED;

    int search_pos;

    unsigned short *bi;
    unsigned short *bo_3, *bo_2, *bo_1;

    //bi = blk_sad + macroblocks * 5 + macroblock_index * 4;
    bi = blk_sad + ((macroblocks + macroblock_index) << 2) + macroblocks;

    // Block type 3: 8x16
    //bo_3 = blk_sad + macroblocks * 3 + macroblock_index * 2;
    bo_3 = blk_sad + ((macroblocks + macroblock_index) << 1) + macroblocks;

    // Block type 5: 8x4
    bo_2 = blk_sad + macroblocks + macroblock_index * 2;

    // Block type 4: 8x8
    bo_1 = blk_sad + macroblock_index;

    for (search_pos = threadIdx.x; search_pos < (MAX_POS+1)/2; search_pos += 32)
    {
        /* Each uint is actually two 2-byte integers packed together.
        * Only addition is used and there is no chance of integer overflow
        * so this can be done to reduce computation time. */
        uint i00 = ((uint *)bi)[search_pos];
        uint i01 = ((uint *)bi)[search_pos + MAX_POS_PADDED/2];
        uint i10 = ((uint *)bi)[search_pos + 2*MAX_POS_PADDED/2];
        uint i11 = ((uint *)bi)[search_pos + 3*MAX_POS_PADDED/2];

        ((uint *)bo_3)[search_pos]                  = i00 + i10;
        ((uint *)bo_3)[search_pos+MAX_POS_PADDED/2] = i01 + i11;
        ((uint *)bo_2)[search_pos]                  = i00 + i01;
        ((uint *)bo_2)[search_pos+MAX_POS_PADDED/2] = i10 + i11;
        ((uint *)bo_1)[search_pos]                  = (i00 + i01) + (i10 + i11);
    }
}


__global__ void ori_mb_sad_calc(unsigned short *blk_sad,
    unsigned short *frame,
    int mb_width,
    int mb_height)
{
    // extern __shared__ unsigned short sad_loc[];
    // extern __shared__ vec8b sad_loc_8b[];
    __shared__ unsigned short sad_loc[1096];
    // __shared__ vec8b sad_loc_8b[THREADS_W * THREADS_H * MAX_POS_PADDED / 8];
    vec8b *sad_loc_8b = (vec8b *) &sad_loc[0];

    int txy_tmp = threadIdx.x / CEIL(MAX_POS, POS_PER_THREAD);
    int ty = txy_tmp / THREADS_W;
    int tx = txy_tmp - __umul24(ty, THREADS_W);
    int bx = blockIdx.x;
    int by = blockIdx.y;

    /* Macroblock and sub-block coordinates */
    int mb_x = (tx + __umul24(bx, THREADS_W)) >> 2;
    int mb_y = (ty + __umul24(by, THREADS_H)) >> 2;
    int block_x = (tx + __umul24(bx, THREADS_W)) & 0x03;
    int block_y = (ty + __umul24(by, THREADS_H)) & 0x03;

    /* Block-copy data into shared memory.
    * Threads are grouped into sets of 16, leaving some threads idle. */
    if ((threadIdx.x >> 4) < (THREADS_W * THREADS_H))
    {
        int ty = (threadIdx.x >> 4) / THREADS_W;
        int tx = (threadIdx.x >> 4) - __umul24(ty, THREADS_W);
        int tgroup = threadIdx.x & 15;

        /* Width of the image in pixels */
        int img_width = mb_width*16;

        /* Pixel offset of the origin of the current 4x4 block */
        int frame_x = (tx + __umul24(bx, THREADS_W)) << 2;
        int frame_y = (ty + __umul24(by, THREADS_H)) << 2;

        /* Origin in the current frame for this 4x4 block */
        int cur_o = frame_y * img_width + frame_x;

        /* If this is an invalid 4x4 block, do nothing */
        if (((frame_x >> 4) < mb_width) && ((frame_y >> 4) < mb_height))
        {
            /* Copy one pixel into 'frame' */
            FRAME_PUT_1(__umul24(ty, THREADS_W) + tx, tgroup,
            frame[cur_o + (tgroup >> 2) * img_width + (tgroup & 3)]);
        }
    }

    __syncthreads();

    /* If this thread is assigned to an invalid 4x4 block, do nothing */
    if ((mb_x < mb_width) && (mb_y < mb_height))
    {
        /* Pixel offset of the origin of the current 4x4 block */
        int frame_x = ((mb_x << 2) + block_x) << 2;
        int frame_y = ((mb_y << 2) + block_y) << 2;

        /* Origin of the search area for this 4x4 block */
        int ref_x = frame_x - SEARCH_RANGE;
        int ref_y = frame_y - SEARCH_RANGE;

        /* Origin in the current frame for this 4x4 block */
        int cur_o = ty * THREADS_W + tx;

        int search_pos;
        int search_pos_base =
        (threadIdx.x % CEIL(MAX_POS, POS_PER_THREAD)) * POS_PER_THREAD;
        int search_pos_end = search_pos_base + POS_PER_THREAD;

        int sotmp = search_pos_base / SEARCH_DIMENSION;
        int local_search_off_x = search_pos_base - TIMES_DIM_POS(sotmp);
        int search_off_y = ref_y + sotmp;

        /* Don't go past bounds */
        if (search_pos_end > MAX_POS) {
            search_pos_end = MAX_POS;
        }

        /* For each search position, within the range allocated to this thread */
        for (search_pos = search_pos_base;
            search_pos < search_pos_end;
            search_pos += 3) {
            /* It is also beneficial to fuse (jam) the enclosed loops if this loop
            * is unrolled. */
            unsigned short sad1 = 0, sad2 = 0, sad3 = 0;
            int search_off_x = ref_x + local_search_off_x;

            /* 4x4 SAD computation */
            for(int y=0; y<4; y++) {
                int t;
                t = tex2D(ref, search_off_x, search_off_y + y);
                sad1 += abs(t - FRAME_GET(cur_o, 0, y));

                t = tex2D(ref, search_off_x + 1, search_off_y + y);
                sad1 += abs(t - FRAME_GET(cur_o, 1, y));
                sad2 += abs(t - FRAME_GET(cur_o, 0, y));

                t = tex2D(ref, search_off_x + 2, search_off_y + y);
                sad1 += abs(t - FRAME_GET(cur_o, 2, y));
                sad2 += abs(t - FRAME_GET(cur_o, 1, y));
                sad3 += abs(t - FRAME_GET(cur_o, 0, y));

                t = tex2D(ref, search_off_x + 3, search_off_y + y);
                sad1 += abs(t - FRAME_GET(cur_o, 3, y));
                sad2 += abs(t - FRAME_GET(cur_o, 2, y));
                sad3 += abs(t - FRAME_GET(cur_o, 1, y));

                t = tex2D(ref, search_off_x + 4, search_off_y + y);
                sad2 += abs(t - FRAME_GET(cur_o, 3, y));
                sad3 += abs(t - FRAME_GET(cur_o, 2, y));

                t = tex2D(ref, search_off_x + 5, search_off_y + y);
                sad3 += abs(t - FRAME_GET(cur_o, 3, y));
            }

            /* Save this value into the local SAD array */
            SAD_LOC_PUT(__umul24(ty, THREADS_W) + tx, search_pos, sad1);
            SAD_LOC_PUT(__umul24(ty, THREADS_W) + tx, search_pos+1, sad2);
            SAD_LOC_PUT(__umul24(ty, THREADS_W) + tx, search_pos+2, sad3);

            local_search_off_x += 3;
            if (local_search_off_x >= SEARCH_DIMENSION)
            {
                local_search_off_x -= SEARCH_DIMENSION;
                search_off_y++;
            }
        }
    }

    __syncthreads();

    /* Block-copy data into global memory.
    * Threads are grouped into sets of 32, leaving some threads idle. */
    if ((threadIdx.x >> 5) < (THREADS_W * THREADS_H))
    {
        int tgroup = threadIdx.x & 31;
        int ty = (threadIdx.x >> 5) / THREADS_W;
        int tx = (threadIdx.x >> 5) - __umul24(ty, THREADS_W);
        int index;

        /* Macroblock and sub-block coordinates */
        int mb_x = (tx + __umul24(bx, THREADS_W)) >> 2;
        int mb_y = (ty + __umul24(by, THREADS_H)) >> 2;
        int block_x = (tx + __umul24(bx, THREADS_W)) & 0x03;
        int block_y = (ty + __umul24(by, THREADS_H)) & 0x03;

        if ((mb_x < mb_width) && (mb_y < mb_height))
        {
            /* All SADs from this thread are stored in a contiguous chunk
            * of memory starting at this offset */
            blk_sad += (__umul24(__umul24(mb_width, mb_height), 25) +
            (__umul24(mb_y, mb_width) + mb_x) * 16 +
            (4 * block_y + block_x)) *
            MAX_POS_PADDED;

            /* Block copy, 32 threads at a time */
            for (index = tgroup; index < SAD_LOC_8B_ROW_SIZE; index += 32)
            ((vec8b *)blk_sad)[index] 
            = SAD_LOC_8B_GET(__umul24(ty, THREADS_W) + tx, index);
        }
    }
}


texture<unsigned short, 2, cudaReadModeElementType> &get_ref(void)
{
  return ref;
}

texture<unsigned short, 2, cudaReadModeElementType> &get_ref_2(void)
{
  return ref_2;
}