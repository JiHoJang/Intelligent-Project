#include <stdint.h>
#include <stdint.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main() {
    int width, height, bpp;

    uint8_t* rgb_image = stbi_load("1.png", &width, &height, &bpp, 3);
	printf("%d %d %d\n",width,height,bpp);
	for(int i=0;i<height;i++){
		for(int j=0;j<width;j++){
			for(int z=0;z<3;z++){
				printf("%u ",rgb_image[width*3*i+3*j+z]);
			}
			printf("\t");
		}
		printf("\n");
	}
    stbi_image_free(rgb_image);
	
    return 0;
}
