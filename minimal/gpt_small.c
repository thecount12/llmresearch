#include <u.h>
#include <libc.h>

// Define a struct with two fields
typedef struct MyData {
    float field1;
    float field2;
} MyData;

// Function to create and initialize the array of structs
MyData*
gpt2_forward(int size) {
    MyData *data_array;

    data_array = malloc(size * sizeof(MyData));
    if (data_array == nil) {
        sysfatal("malloc: %r");
    }

    for (int i = 0; i < size; i++) {
        data_array[i].field1 = (float)i;
        data_array[i].field2 = (float)i * 2.0;
    }
    return data_array;
}

void
main(void) {
    int size = 3;
    MyData *p;

    print("hello\n");
  
    p = gpt2_forward(size);
  
    print("Output:\n");
    for (int i = 0; i < size; i++) {
        print("Data[%d]: field1=%f, field2=%f\n", i, p[i].field1, p[i].field2);
    }
  
    free(p);
  
    exits(nil);
}
