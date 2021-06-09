#FCLAGS  = `pkg-config --cflags neko` -I/cm/shared/opt/intelFPGA_pro/20.2.0/hld/host/include -fallow-invalid-boz
LFLAGS    = -L/cm/shared/opt/intelFPGA_pro/19.4.0/hld/board/bittware_pcie/s10/linux64/lib -L/cm/shared/opt/intelFPGA_pro/20.2.0/hld/host/linux64/lib -lalteracl -lbitt_s10_pcie_mmd -lelf -lOpenCL

FCLAGS  = `pkg-config --cflags neko` -fallow-invalid-boz -lOpenCL -I/cm/shared/opt/intelFPGA_pro/20.2.0/hld/host/include

LIBS    = `pkg-config --libs neko` 
FC      = mpif90

DEST    = poisson
SRC	= setup.f90 clroutines.f90 driver.f90
OBJ	= ${SRC:.f90=.o}

all: $(DEST)

install:

clean:
	-rm -f *.o core *.core $(OBJ) $(DEST)

$(DEST): ${OBJ}
	$(FC) $(FCLAGS) ${OBJ} -o $@  $(LIBS) ${LFLAGS}

%.o: %.f90
	${FC} ${FCLAGS} -c $< clfortran.f90


