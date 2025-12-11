
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "math.h"


//khai báo hàm Prediction
void Prediction(float image[28][28],
                float w_conv1[6][1][1],
                float w_conv2[16][6][5][5],
                float w_fc1[120][400],
                float w_fc2[84][120],
                float w_fc3[10][84],
                float b_conv1[6],
                float b_conv2[16],
                float b_fc1[120],
                float b_fc2[84],
                float b_fc3[10],
                float probs[10]);

int main(int argc, char** argv){

   //float image[28][28];
   float w_conv1[6][1][1];
   float w_conv2[16][6][5][5];
   float w_fc1[120][400];
   float w_fc2[84][120];
   float w_fc3[10][84];
   float b_conv1[6];
   float b_conv2[16];
   float b_fc1[120];
   float b_fc2[84];
   float b_fc3[10];
   float probs[10];

   int i,j,m,n,index;
   FILE *fp;

    /* Load Weights from DDR->LMM */
   fp = fopen("weights\\w_conv1.txt", "r");
   for(i=0;i<6;i++)
       fscanf(fp, "%f ",  &(w_conv1[i][0][0]));  fclose(fp);
/*
   fp = fopen("w_conv2.txt", "r");
   for(i=0;i<16;i++){
       for(j=0;j<6;j++){
           for(m=0;m<5;m++){
               for(n=0;n<5;n++){
                   index = 16*i + 6*j + 5*m + 5*n;
                   fscanf(fp, "%f ",  &(w_conv2[i][j][m][n]));
               }
           }
       }
   }
   fclose(fp);
*/
    fp = fopen("weights\\w_conv2.txt", "r");
   for(i = 0; i < 16; i++) {
       for(j = 0; j < 6; j++) {
           for(m = 0; m < 5; m++) {
               for(n = 0; n < 5; n++) {
                   fscanf(fp, "%f", &w_conv2[i][j][m][n]);
               }
           }
       }
   }
   fclose(fp);

   fp = fopen("weights\\w_fc1.txt", "r");
   for(i=0;i<120;i++){
       for(j=0;j<400;j++)
           fscanf(fp, "%f ",  &(w_fc1[i][j]));
   }
   fclose(fp);

   fp = fopen("weights/w_fc2.txt", "r");
   for(i=0;i<84;i++){
       for(j=0;j<120;j++)
           fscanf(fp, "%f ",  &(w_fc2[i][j]));
   }
   fclose(fp);

   fp = fopen("weights/w_fc3.txt", "r");
   for(i=0;i<10;i++){
       for(j=0;j<84;j++)
           fscanf(fp, "%f ",  &(w_fc3[i][j]));
   }
   fclose(fp);

   fp = fopen("weights/b_conv1.txt", "r");
   for(i=0;i<6;i++)
       fscanf(fp, "%f ",  &(b_conv1[i]));  fclose(fp);

   fp = fopen("weights/b_conv2.txt", "r");
   for(i=0;i<16;i++)
       fscanf(fp, "%f ",  &(b_conv2[i]));  fclose(fp);

   fp = fopen("weights/b_fc1.txt", "r");
   for(i=0;i<120;i++)
       fscanf(fp, "%f ",  &(b_fc1[i]));  fclose(fp);

   fp = fopen("weights/b_fc2.txt", "r");
   for(i=0;i<84;i++)
       fscanf(fp, "%f ",  &(b_fc2[i]));  fclose(fp);

   fp = fopen("weights/b_fc3.txt", "r");
   for(i=0;i<10;i++)
       fscanf(fp, "%f ",  &(b_fc3[i]));  fclose(fp);

   float *dataset = (float*)malloc(LABEL_LEN*28*28 *sizeof(float));
   int target[LABEL_LEN];

   fp = fopen("mnist-test-target.txt", "r");
   for(i=0;i<LABEL_LEN;i++)
       fscanf(fp, "%d ",  &(target[i]));  fclose(fp);

   fp = fopen("mnist-test-image.txt", "r");
   for(i=0;i<LABEL_LEN*28*28;i++)
       fscanf(fp, "%f ",  &(dataset[i]));  fclose(fp);

   float image[28][28];
   float *datain;
   int acc = 0;
   int mm, nn;
   for(i=0;i<LABEL_LEN;i++) {

       datain = &dataset[i*28*28];
       for(mm=0;mm<28;mm++)
           for(nn=0;nn<28;nn++)
               image[mm][nn] = *(float*)&datain[28*mm + nn];

       Prediction(   image,
                     w_conv1,
                     w_conv2,
                     w_fc1,
                     w_fc2,
                     w_fc3,
                     b_conv1,
                     b_conv2,
                     b_fc1,
                     b_fc2,
                     b_fc3,
                     probs
                     );

       int index = 0;
       float max = probs[0];
       for (j=1;j<10;j++) {
            if (probs[j] > max) {
                index = j;
                max = probs[j];
            }
       }

       if (index == target[i]) acc++;
       printf("Predicted label: %d\n", index);
       printf("Prediction: %d/%d\n", acc, i+1);
   }
   printf("Accuracy = %f\n", acc*1.0f/LABEL_LEN);

    return 0;
}



// ĐỊNH NGHĨA HÀM Prediction 
void Prediction(float image[28][28],
                float w_conv1[6][1][1],
                float w_conv2[16][6][5][5],
                float w_fc1[120][400],
                float w_fc2[84][120],
                float w_fc3[10][84],
                float b_conv1[6],
                float b_conv2[16],
                float b_fc1[120],
                float b_fc2[84],
                float b_fc3[10],
                float probs[10]) {

    float conv1_out[6][28][28];
    float pool1_out[6][14][14];
    float conv2_out[16][10][10];
    float pool2_out[16][5][5];
    float flatten[400];
    float fc1_out[120];
    float fc2_out[84];
    float fc3_out[10];

    int i,j,k,l,c,ii,jj;

    // Conv1 + ReLU
    for(c=0; c<6; c++) {
        for(i=0; i<28; i++) {
            for(j=0; j<28; j++) {
                float val = b_conv1[c] + image[i][j] * w_conv1[c][0][0];
                conv1_out[c][i][j] = val > 0 ? val : 0;
            }
        }
    }

    // Pool1
    for(c=0; c<6; c++) for(i=0; i<14; i++) for(j=0; j<14; j++) {
        float s = conv1_out[c][i*2][j*2] + conv1_out[c][i*2][j*2+1] +
                  conv1_out[c][i*2+1][j*2] + conv1_out[c][i*2+1][j*2+1];
        pool1_out[c][i][j] = s / 4.0;
    }

    // Conv2 + ReLU
    for(c=0; c<16; c++) for(ii=0; ii<10; ii++) for(jj=0; jj<10; jj++) {
        float sum = b_conv2[c];
        for(int ich=0; ich<6; ich++) for(k=0; k<5; k++) for(l=0; l<5; l++) {
            sum += pool1_out[ich][ii+k][jj+l] * w_conv2[c][ich][k][l];
        }
        conv2_out[c][ii][jj] = sum > 0 ? sum : 0;
    }

    // Pool2
    for(c=0; c<16; c++) for(i=0; i<5; i++) for(j=0; j<5; j++) {
        float s = conv2_out[c][i*2][j*2] + conv2_out[c][i*2][j*2+1] +
                  conv2_out[c][i*2+1][j*2] + conv2_out[c][i*2+1][j*2+1];
        pool2_out[c][i][j] = s / 4.0;
    }

    // Flatten
    int idx = 0;
    for(c=0; c<16; c++) for(i=0; i<5; i++) for(j=0; j<5; j++)
        flatten[idx++] = pool2_out[c][i][j];

    // FC1 + ReLU
    for(i=0; i<120; i++) {
        float sum = b_fc1[i];
        for(j=0; j<400; j++) sum += flatten[j] * w_fc1[i][j];
        fc1_out[i] = sum > 0 ? sum : 0;
    }

    // FC2 + ReLU
    for(i=0; i<84; i++) {
        float sum = b_fc2[i];
        for(j=0; j<120; j++) sum += fc1_out[j] * w_fc2[i][j];
        fc2_out[i] = sum > 0 ? sum : 0;
    }

    // FC3
    for(i=0; i<10; i++) {
        float sum = b_fc3[i];
        for(j=0; j<84; j++) sum += fc2_out[j] * w_fc3[i][j];
        fc3_out[i] = sum;
    }

    // Log Softmax
    float maxv = fc3_out[0];
    for(i=1; i<10; i++) if(fc3_out[i] > maxv) maxv = fc3_out[i];

    float sum_exp = 0.0f;
    for(i=0; i<10; i++) {
        float val = expf(fc3_out[i] - maxv);  // dùng expf cho float
        probs[i] = val;
        sum_exp += val;
    }
    for(i=0; i<10; i++) {
        probs[i] = logf(probs[i] / sum_exp);  // logf cho float
    }
}