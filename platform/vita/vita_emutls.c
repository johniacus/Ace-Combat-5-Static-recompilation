/* Thread-local storage for the Vita build.
 *
 * GCC has no native TLS on arm-vita-eabi, so every __thread / _Thread_local
 * access compiles to a call to __emutls_get_address(). The libgcc version of
 * that function is unusable here, in two different ways:
 *
 *  - it decides whether the program is threaded by testing whether
 *    pthread_cancel is linked. The Vita's pthread library has none, so libgcc
 *    takes its single-threaded path and every thread shares one copy of each
 *    "thread-local" variable;
 *  - if that test is forced true, its static lock was initialised for a
 *    different pthread ABI (zero rather than the Vita library's autoinit
 *    value), and the first unlock dereferences NULL (crash dumps,
 *    __gthread_mutex_unlock -> pthread_mutex_unlock).
 *
 * Defining both emutls entry points here keeps libgcc's emutls.o out of the
 * link. Same ABI and behaviour as libgcc, built on the Vita pthread API. */
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* libgcc uses mode(word): 32-bit on the Vita, 64-bit on x86-64 hosts. */
typedef uintptr_t emu_word;

typedef struct emutls_object {
    emu_word size;
    emu_word align;
    union {
        uintptr_t offset;   /* 1-based slot index once assigned, 0 before */
        void *ptr;
    } loc;
    void *templ;            /* initial value, or NULL for zero-fill */
} emutls_object;

typedef struct emutls_array {
    uintptr_t size;         /* number of slots in data[] */
    void *data[];
} emutls_array;

static pthread_mutex_t emutls_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t emutls_once = PTHREAD_ONCE_INIT;
static pthread_key_t emutls_key;
static uintptr_t emutls_count;

static void emutls_destroy(void *p) {
    emutls_array *arr = (emutls_array *)p;
    if (!arr) return;
    for (uintptr_t i = 0; i < arr->size; i++)
        if (arr->data[i])
            free(((void **)arr->data[i])[-1]);
    free(arr);
}

static void emutls_init(void) {
    pthread_key_create(&emutls_key, emutls_destroy);
}

static void *emutls_alloc(emutls_object *obj) {
    uintptr_t align = obj->align < sizeof(void *) ? sizeof(void *) : obj->align;
    /* One pointer before the aligned block remembers what malloc returned. */
    char *raw = (char *)malloc(obj->size + align + sizeof(void *));
    if (!raw) abort();
    uintptr_t at = ((uintptr_t)raw + sizeof(void *) + align - 1) & ~(align - 1);
    void *ret = (void *)at;
    ((void **)ret)[-1] = raw;
    if (obj->templ)
        memcpy(ret, obj->templ, obj->size);
    else
        memset(ret, 0, obj->size);
    return ret;
}

void *__emutls_get_address(void *vobj) {
    emutls_object *obj = (emutls_object *)vobj;
    uintptr_t offset = __atomic_load_n(&obj->loc.offset, __ATOMIC_ACQUIRE);

    if (__builtin_expect(offset == 0, 0)) {
        pthread_once(&emutls_once, emutls_init);
        pthread_mutex_lock(&emutls_mutex);
        offset = obj->loc.offset;
        if (offset == 0) {
            offset = ++emutls_count;
            __atomic_store_n(&obj->loc.offset, offset, __ATOMIC_RELEASE);
        }
        pthread_mutex_unlock(&emutls_mutex);
    }

    emutls_array *arr = (emutls_array *)pthread_getspecific(emutls_key);
    if (__builtin_expect(arr == NULL || offset > arr->size, 0)) {
        uintptr_t old = arr ? arr->size : 0;
        uintptr_t want = offset + 32;
        emutls_array *grown = (emutls_array *)realloc(
            arr, sizeof(emutls_array) + want * sizeof(void *));
        if (!grown) abort();
        memset(grown->data + old, 0, (want - old) * sizeof(void *));
        grown->size = want;
        arr = grown;
        pthread_setspecific(emutls_key, arr);
    }

    void *ret = arr->data[offset - 1];
    if (__builtin_expect(ret == NULL, 0)) {
        ret = emutls_alloc(obj);
        arr->data[offset - 1] = ret;
    }
    return ret;
}

void __emutls_register_common(void *vobj, unsigned int size,
                              unsigned int align, void *templ) {
    emutls_object *obj = (emutls_object *)vobj;
    if (obj->size < size) {
        obj->size = size;
        obj->templ = NULL;
    }
    if (obj->align < align)
        obj->align = align;
    if (templ && size == obj->size)
        obj->templ = templ;
}
