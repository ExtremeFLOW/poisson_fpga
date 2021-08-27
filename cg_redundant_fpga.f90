!> Defines various Conjugate Gradient methods
module cg_fpga
  use krylov
  use math
  use num_types
  use fast3d
  use tensor
  use clroutines
  use iso_c_binding
  implicit none

  !> FPGA unpreconditioned conjugate gradient method
  type, public, extends(ksp_t) :: cg_fpga_t
     real(kind=rp), allocatable :: w(:)
     real(kind=rp), allocatable :: r(:)
     real(kind=rp), allocatable :: p(:)
     integer(c_intptr_t) :: cmd_queue, cl_cg_kernel, context
     integer(c_intptr_t) :: cl_x, cl_w, cl_p, cl_res
     integer(c_intptr_t) :: cl_x_cord, cl_y_cord, cl_z_cord, cl_jx, cl_jxt, cl_w3, cl_jacinv
     integer(c_intptr_t) :: cl_mult, cl_dx, cl_dxt, cl_rtz1, cl_rtz2, cl_beta, cl_mask
     integer(c_intptr_t) :: cl_b, cl_dg, cl_gd
     integer(c_size_t) :: real_size = rp
     integer(c_size_t) :: array_size, element_size, dx_size
     integer :: n, lo, nb, gs_m
 
   contains
     procedure, pass(this) :: init => cg_fpga_init
     procedure, pass(this) :: free => cg_fpga_free
     procedure, pass(this) :: solve => cg_fpga_solve
  end type cg_fpga_t

contains

  !> Initialise a standard PCG solver
  subroutine cg_fpga_init(this, n, M, rel_tol, abs_tol)
    class(cg_fpga_t), intent(inout) :: this
    class(pc_t), optional, intent(inout), target :: M
    integer, intent(in) :: n
    real(kind=rp), optional, intent(inout) :: rel_tol
    real(kind=rp), optional, intent(inout) :: abs_tol

        
    call this%free()
    
    allocate(this%w(n))
    allocate(this%r(n))
    allocate(this%p(n))

    if (present(rel_tol) .and. present(abs_tol)) then
       call this%ksp_init(rel_tol, abs_tol)
    else if (present(rel_tol)) then
       call this%ksp_init(rel_tol=rel_tol)
    else if (present(abs_tol)) then
       call this%ksp_init(abs_tol=abs_tol)
    else
       call this%ksp_init()
    end if
    !> Size of arrays in bytes
    this%array_size= rp*n
    this%n = n
          
  end subroutine cg_fpga_init

  subroutine cg_fpga_init_device(this,idevice, iplatform, bitstream, lx)
    class(cg_fpga_t), target :: this
    integer :: idevice, iplatform
    character(len=80) :: bitstream
    integer(c_int32_t) :: err
    character(len=1024) :: kernel_name
    integer(c_size_t),target ::  length
    integer(c_intptr_t), target :: prog
    integer(c_intptr_t), allocatable, target :: platform_ids(:), device_ids(:)
    integer :: irec, i
    integer :: filesize , lx
    integer, target :: zero =0
    character(len=1,kind=c_char), allocatable, target :: binary(:)
    character(len=1,kind=c_char), target :: c_kernel_name(1:1024)
    type(c_ptr), target :: psource  

    this%dx_size = rp*lx**2
    call create_device_context(iplatform, platform_ids,&
                               idevice, device_ids, this%context, this%cmd_queue)
    call query_platform_info(platform_ids(iplatform))
    call read_file(bitstream,binary,filesize)
    length = filesize
    psource = C_LOC(binary)
    prog = clCreateProgramWithBinary(this%context, 1, C_LOC(device_ids(idevice)),&
                                     C_LOC(length),C_LOC(psource),C_NULL_PTR,err)
    if (err .ne. 0) stop ' prog binary'
    
    err=clBuildProgram(prog, 0, C_NULL_PTR,C_NULL_PTR,C_NULL_FUNPTR,C_NULL_PTR)
    if (err .ne. 0) stop 'not created prog'
    
    kernel_name = "cg"
    irec=len(trim(kernel_name))
    do i=1,irec
       c_kernel_name(i)=kernel_name(i:i)
    enddo
    c_kernel_name(irec+1)=C_NULL_CHAR
    this%cl_cg_kernel=clCreateKernel(prog,C_LOC(c_kernel_name),err)
    if (err.ne.0) stop 'clCreateKernel'
      
 

    err = clReleaseProgram(prog)
    if (err.ne.0) stop 'clReleaseProgram'
    this%cl_x = clCreateBuffer(this%context,ior(CL_MEM_READ_WRITE,CL_CHANNEL_4_INTELFPGA),&
                          this%array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_w = clCreateBuffer(this%context,ior(CL_MEM_READ_WRITE, CL_CHANNEL_1_INTELFPGA),&
                          this%array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_res = clCreateBuffer(this%context,ior(CL_MEM_READ_WRITE,CL_CHANNEL_1_INTELFPGA),&
                          this%array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_p = clCreateBuffer(this%context,ior(CL_MEM_READ_WRITE,CL_CHANNEL_2_INTELFPGA),&
                          this%array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'

    this%cl_mult = clCreateBuffer(this%context,ior(CL_MEM_READ_ONLY,CL_CHANNEL_1_INTELFPGA),&
                          this%array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_x_cord = clCreateBuffer(this%context,&
                           ior(CL_MEM_READ_ONLY, CL_CHANNEL_4_INTELFPGA),&
                           this%array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_y_cord = clCreateBuffer(this%context,&
                           ior(CL_MEM_READ_ONLY, CL_CHANNEL_4_INTELFPGA),&
                           this%array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_z_cord = clCreateBuffer(this%context,&
                           ior(CL_MEM_READ_ONLY, CL_CHANNEL_4_INTELFPGA),&
                           this%array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_jx = clCreateBuffer(this%context,&
                           ior(CL_MEM_READ_ONLY, CL_CHANNEL_2_INTELFPGA),&
                           this%array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_jxt = clCreateBuffer(this%context,&
                           ior(CL_MEM_READ_ONLY, CL_CHANNEL_3_INTELFPGA),&
                           this%array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_w3 = clCreateBuffer(this%context,& 
                           ior(CL_MEM_READ_ONLY, CL_CHANNEL_4_INTELFPGA),&
                           this%array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
 
    this%cl_dx = clCreateBuffer(this%context,&
                           ior(CL_MEM_READ_ONLY, CL_CHANNEL_1_INTELFPGA),&
                           this%dx_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_dxt = clCreateBuffer(this%context,&
                            ior(CL_MEM_READ_ONLY, CL_CHANNEL_2_INTELFPGA),&
                            this%dx_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_rtz1 = clCreateBuffer(this%context,& 
                             CL_MEM_READ_WRITE,int(8,8),C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_rtz2 = clCreateBuffer(this%context,& 
                             CL_MEM_READ_WRITE,int(8,8),C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_beta = clCreateBuffer(this%context,& 
                             CL_MEM_READ_WRITE,int(8,8),C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_mask = clCreateBuffer(this%context,& 
                             CL_MEM_READ_WRITE,this%array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_b = clCreateBuffer(this%context,ior(CL_MEM_READ_WRITE,CL_CHANNEL_1_INTELFPGA),&
                          this%array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_gd = clCreateBuffer(this%context,ior(CL_MEM_READ_WRITE,CL_CHANNEL_4_INTELFPGA),&
                          this%array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_dg = clCreateBuffer(this%context,ior(CL_MEM_READ_WRITE,CL_CHANNEL_4_INTELFPGA),&
                          this%array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    this%cl_jacinv = clCreateBuffer(this%context,ior(CL_MEM_READ_WRITE,CL_CHANNEL_3_INTELFPGA),&
                          this%array_size,C_NULL_PTR, err)
    if (err.ne.0) stop 'clCreateBuffer'
    
    err=clSetKernelArg(this%cl_cg_kernel,0,sizeof(this%cl_x),C_LOC(this%cl_x))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,1,sizeof(this%cl_p),C_LOC(this%cl_p))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,2,sizeof(this%cl_res),C_LOC(this%cl_res))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,3,sizeof(this%cl_w),C_LOC(this%cl_w))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,4,sizeof(this%cl_mult),C_LOC(this%cl_mult))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,5,sizeof(this%cl_x_cord),C_LOC(this%cl_x_cord))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,6,sizeof(this%cl_y_cord),C_LOC(this%cl_y_cord))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,7,sizeof(this%cl_z_cord),C_LOC(this%cl_z_cord))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,8,sizeof(this%cl_jx),C_LOC(this%cl_jx))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,9,sizeof(this%cl_jxt),C_LOC(this%cl_jxt))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,10,sizeof(this%cl_w3),C_LOC(this%cl_w3))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,11,sizeof(this%cl_dx),C_LOC(this%cl_dx))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,12,sizeof(this%cl_dxt),C_LOC(this%cl_dxt))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,13,sizeof(this%cl_mask),C_LOC(this%cl_mask))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,14,sizeof(this%cl_rtz1),C_LOC(this%cl_rtz1))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,15,sizeof(this%cl_rtz2),C_LOC(this%cl_rtz2))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,16,sizeof(this%cl_beta),C_LOC(this%cl_beta))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,17,sizeof(this%n),C_LOC(this%n))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,18,sizeof(this%cl_b),C_LOC(this%cl_b))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,19,sizeof(this%cl_gd),C_LOC(this%cl_gd))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,20,sizeof(this%cl_dg),C_LOC(this%cl_dg))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,21,sizeof(this%cl_jacinv),C_LOC(this%cl_jacinv))
    if (err.ne.0) stop 'clSetKernelArg'

  end subroutine cg_fpga_init_device


  subroutine cg_fpga_populate(this, r, w, x, Xh,  c_Xh, bclst, gs_Xh, rtz1, beta)
    class(cg_fpga_t), target :: this
    type(field_t), target :: r, w, x
    type(space_t), target :: Xh
    type(coef_t), target :: c_Xh
    type(bc_list_t) :: bclst
    type(gs_t), target :: gs_Xh
    real(kind=rp), target :: rtz1, beta
    integer :: err, e, j, k, lx, ly, lz
    real(kind=rp) :: zgml(Xh%lx, 3)
    real(kind=rp), target :: jx(Xh%lx*2), jy(Xh%lx*2), jz(Xh%lx*2)
    real(kind=rp), target :: jxt(Xh%lx*2), jyt(Xh%lx*2), jzt(Xh%lx*2)
    real(kind=rp), dimension(2), parameter :: zlin = (/-1d0, 1d0/)
    real(kind=rp), allocatable, target :: x_cord(:,:), y_cord(:,:), z_cord(:,:)
    
    this%gs_m = gs_Xh%nlocal
    this%lo = gs_Xh%local_facet_offset
    this%nb = gs_Xh%nlocal_blks
    !this%gs_m = 0
    !this%lo = 1
    !this%nb = 0
    print *, 'nlocal unique ids', this%gs_m, 'local dofs', this%lo,&
           'number of blocks', this%nb
    err=clSetKernelArg(this%cl_cg_kernel,22,sizeof(this%gs_m),C_LOC(this%gs_m))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,23,sizeof(this%lo),C_LOC(this%lo))
    if (err.ne.0) stop 'clSetKernelArg'
    err=clSetKernelArg(this%cl_cg_kernel,24,sizeof(this%nb),C_LOC(this%nb))
    if (err.ne.0) stop 'clSetKernelArg'
    lx = Xh%lx
    ly = Xh%ly
    lz = Xh%lz
    
    allocate(x_cord(8,c_Xh%msh%nelv), y_cord(8, c_Xh%msh%nelv), z_cord(8, c_Xh%msh%nelv))
    do e = 1, c_Xh%msh%nelv
       x_cord(1,e) = c_Xh%dof%x(1,1,1,e)
       x_cord(2,e) = c_Xh%dof%x(lx,1,1,e)
       x_cord(3,e) = c_Xh%dof%x(1,ly,1,e)
       x_cord(4,e) = c_Xh%dof%x(lx,ly,1,e)

       x_cord(5,e) = c_Xh%dof%x(1,1,lz,e)
       x_cord(6,e) = c_Xh%dof%x(lx,1,lz,e)
       x_cord(7,e) = c_Xh%dof%x(1,ly,lz,e)
       x_cord(8,e) = c_Xh%dof%x(lx,ly,lz,e)
       
       y_cord(1,e) = c_Xh%dof%y(1,1,1,e)
       y_cord(2,e) = c_Xh%dof%y(lx,1,1,e)
       y_cord(3,e) = c_Xh%dof%y(1,ly,1,e)
       y_cord(4,e) = c_Xh%dof%y(lx,ly,1,e)

       y_cord(5,e) = c_Xh%dof%y(1,1,lz,e)
       y_cord(6,e) = c_Xh%dof%y(lx,1,lz,e)
       y_cord(7,e) = c_Xh%dof%y(1,ly,lz,e)
       y_cord(8,e) = c_Xh%dof%y(lx,ly,lz,e)
       
       z_cord(1,e) = c_Xh%dof%z(1,1,1,e)
       z_cord(2,e) = c_Xh%dof%z(lx,1,1,e)
       z_cord(3,e) = c_Xh%dof%z(1,ly,1,e)
       z_cord(4,e) = c_Xh%dof%z(lx,ly,1,e)

       z_cord(5,e) = c_Xh%dof%z(1,1,lz,e)
       z_cord(6,e) = c_Xh%dof%z(lx,1,lz,e)
       z_cord(7,e) = c_Xh%dof%z(1,ly,lz,e)
       z_cord(8,e) = c_Xh%dof%z(lx,ly,lz,e)
    end do
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_x_cord,CL_TRUE,0_8,int(rp*c_Xh%msh%nelv*8,8),&
                               C_LOC(x_cord), 0,C_NULL_PTR,C_NULL_PTR)
    if (err .ne. 0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_y_cord,CL_TRUE,0_8,int(rp*c_Xh%msh%nelv*8,8),&
                               C_LOC(y_cord), 0,C_NULL_PTR,C_NULL_PTR)
    if (err .ne. 0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_z_cord,CL_TRUE,0_8,int(rp*c_Xh%msh%nelv*8,8),&
                               C_LOC(z_cord), 0,C_NULL_PTR,C_NULL_PTR)
    if (err .ne. 0) stop 'clEnqueueWriteBuffer'
    deallocate(x_cord, y_cord, z_cord)
    
    call copy(zgml(1,1), Xh%zg(1,1), Xh%lx)                               
    call copy(zgml(1,2), Xh%zg(1,2), Xh%ly)
    call copy(zgml(1,3), Xh%zg(1,3), Xh%lz)
    
    k = 1
    do j = 1, Xh%lx
       call fd_weights_full(zgml(j,1),zlin,1,0,jxt(k))
       call fd_weights_full(zgml(j,2),zlin,1,0,jyt(k))
       call fd_weights_full(zgml(j,3),zlin,1,0,jzt(k))
       k = k + 2
    end do
    print *, jxt
    print *, jyt
    print *, jzt
   
    call trsp(jx, Xh%lx, jxt, 2)
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_jx,CL_TRUE,0_8,int(rp*lx*2,8),&
                               C_LOC(jx), 0,C_NULL_PTR,C_NULL_PTR)
    if (err .ne. 0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_jxt,CL_TRUE,0_8,int(rp*lx*2,8),&
                               C_LOC(jxt), 0,C_NULL_PTR,C_NULL_PTR)
    if (err .ne. 0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_w3,CL_TRUE,0_8,int(rp*Xh%lxyz,8),&
                               C_LOC(Xh%w3), 0,C_NULL_PTR,C_NULL_PTR)
    if (err .ne. 0) stop 'clEnqueueWriteBuffer'

    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_dx,CL_TRUE,0_8,this%dx_size,&
                               C_LOC(Xh%dx), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_dxt,CL_TRUE,0_8,this%dx_size,&
                               C_LOC(Xh%dxt), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_w,CL_TRUE,0_8,this%array_size,&
                               C_LOC(w%x), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_mult,CL_TRUE,0_8,this%array_size,&
                               C_LOC(c_Xh%mult), 0,C_NULL_PTR,C_NULL_PTR)
    if (err .ne. 0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_mask,CL_TRUE,0_8,int((bclst%bc(1)%bcp%msk(0)+1)*4,8),&
                               C_LOC(bclst%bc(1)%bcp%msk(0)), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_x,CL_TRUE,0_8,this%array_size,&
                               C_LOC(x%x), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_res,CL_TRUE,0_8,this%array_size,&
                               C_LOC(r%x), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_p,CL_TRUE,0_8,this%array_size,&
                               C_LOC(r%x), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_beta,CL_TRUE,0_8,int(rp,8),&
                               C_LOC(beta), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_rtz1,CL_TRUE,0_8,int(rp,8),&
                               C_LOC(rtz1), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_jacinv,CL_TRUE,0_8,this%array_size,&
                               C_LOC(c_Xh%jacinv), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_dg,CL_TRUE,0_8,int(this%gs_m*4,8),&
                               C_LOC(gs_Xh%local_dof_gs), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_gd,CL_TRUE,0_8,int(this%gs_m*4,8),&
                               C_LOC(gs_Xh%local_gs_dof), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer'
    err = clEnqueueWriteBuffer(this%cmd_queue,this%cl_b,CL_TRUE,0_8,int(this%nb*4,8),&
                               C_LOC(gs_Xh%local_blk_len), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer'

    err=clFinish(this%cmd_queue)
    if (err.ne.0) stop 'clFinish'

  end subroutine cg_fpga_populate 
  
  !> Deallocate a standard PCG solver
  subroutine cg_fpga_free(this)
    class(cg_fpga_t), intent(inout) :: this
    integer :: err

    call this%ksp_free()

    if (allocated(this%w)) then
       deallocate(this%w)
    end if

    if (allocated(this%r)) then
       deallocate(this%r)
    end if

    if (allocated(this%p)) then
       deallocate(this%p)
       err=clFinish(this%cmd_queue)
       if (err.ne.0) stop 'clFinish'
       err = clReleaseCommandQueue(this%cmd_queue) 
       err = clReleaseContext(this%context)  
       err = clReleaseMemObject(this%cl_x)
       if (err.ne.0) stop 'clRelasseMemObj'
    end if
   

  end subroutine cg_fpga_free
  
  !> Standard Unpreconditioned CG solve
  function cg_fpga_solve(this, Ax, x, f, n, coef, blst, gs_h, niter) result(ksp_results)
    class(cg_fpga_t), intent(inout):: this
    class(ax_t), intent(inout) :: Ax
    type(field_t), intent(inout) :: x
    integer, intent(inout) :: n
    real(kind=rp), dimension(n), intent(inout) :: f
    type(coef_t), intent(inout) :: coef
    type(bc_list_t), intent(inout) :: blst
    type(gs_t), intent(inout) :: gs_h
    type(ksp_monitor_t) :: ksp_results
    integer, optional, intent(in) :: niter
    real(kind=rp), parameter :: one = 1.0
    real(kind=rp), parameter :: zero = 0.0
    integer :: iter, max_iter, err
    real(kind=rp), target :: rnorm, rtr, rtr0, rtz2, rtz1
    real(kind=rp), target :: beta, pap, alpha, alphm, eps, norm_fac
    
    if (present(niter)) then
       max_iter = niter
    else
       max_iter = KSP_MAX_ITER
    end if
    norm_fac = one/sqrt(coef%volume)

    call rzero(x%x, n)
    call rzero(this%p, n)
    call copy(this%r, f, n)

    rtr = glsc3(this%r, coef%mult, this%r, n)
    print *, 'first',sqrt(rtr)
    rnorm = sqrt(rtr)*norm_fac
    ksp_results%res_start = rnorm
    ksp_results%res_final = rnorm
    ksp_results%iter = 0
    if(rnorm .eq. zero) return
    do iter = 1, max_iter
       err=clEnqueueTask(this%cmd_queue,this%cl_cg_kernel,0,C_NULL_PTR,C_NULL_PTR)
       if (err .ne. 0) stop 'clEnqueueEnqueueTask cg'
       call host_kernel(this, gs_h, blst, n)
       print *,'true after', sqrt(glsc3(this%r, coef%mult, this%r,n))
    end do
    err=clFinish(this%cmd_queue)
    ksp_results%res_final = rnorm
    ksp_results%iter = iter
  end function cg_fpga_solve
 
  subroutine host_kernel(this, gs_h, blst, n)
    class(cg_fpga_t), target :: this
    type(gs_t) :: gs_h
    type(bc_list_t) :: blst
    integer :: n, err
    real(kind=rp), target :: rtz1,rtz2, rnorm
    err = clEnqueueReadBuffer(this%cmd_queue,this%cl_rtz1,CL_TRUE,&
                                 0_8,int(rp,8),C_LOC(rtz1),0,C_NULL_PTR,C_NULL_PTR)
    !err = clEnqueueReadBuffer(this%cmd_queue,this%cl_rtz2,CL_TRUE,&
    !                             0_8,int(rp,8),C_LOC(rtz2),0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueReadBuffer'
    if (err.ne.0) stop 'clEnqueueReadBuffer'
    !err = clEnqueueReadBuffer(this%cmd_queue,this%cl_res,CL_TRUE,&
    !                             0_8,this%array_size,C_LOC(this%r),0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueReadBuffer'
    err=clFinish(this%cmd_queue)
    
    rnorm = sqrt(rtz1)
    print *, 'actual result',rnorm, rtz1
    !call gs_op_vector(gs_h, this%w, n, GS_OP_ADD)
    !err=clEnqueueWriteBuffer(this%cmd_queue,this%cl_w,CL_TRUE,0_8,&
    !                         this%array_size,C_LOC(this%w), 0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueWriteBuffer cg'    
  end subroutine host_kernel
  subroutine fpga_get_data(this, x)
    type(cg_fpga_t), target :: this
    type(field_t), target :: x
    integer :: err
    err = clEnqueueReadBuffer(this%cmd_queue,this%cl_x,CL_TRUE,&
                                 0_8,this%array_size,C_LOC(x%x),0,C_NULL_PTR,C_NULL_PTR)
    if (err.ne.0) stop 'clEnqueueReadBuffer'
   end subroutine fpga_get_data
end module cg_fpga
  

