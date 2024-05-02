// The following ifdef block is the standard way of creating macros which make exporting
// from a DLL simpler. All files within this DLL are compiled with the MOSTDIFFVAL_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see
// MOSTDIFFVAL_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef MOSTDIFFVAL_EXPORTS
#define MOSTDIFFVAL_API __declspec(dllexport)
#else
#define MOSTDIFFVAL_API __declspec(dllimport)
#endif

// This class is exported from the dll
class MOSTDIFFVAL_API CMostDiffVal {
public:
	CMostDiffVal(void);
	// TODO: add your methods here.
};

extern MOSTDIFFVAL_API int nMostDiffVal;

MOSTDIFFVAL_API int fnMostDiffVal(void);
