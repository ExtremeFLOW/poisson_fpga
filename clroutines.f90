module clroutines
    use clfortran
    use ISO_C_BINDING
    implicit none

contains
    subroutine init_kernel(kernel, context, iplatform, platform_ids, idevice, device_ids, kernel_name, source, options)
        integer(c_int32_t) :: ierr
        integer(c_intptr_t), allocatable, target :: platform_ids(:)
        integer(c_intptr_t), allocatable, target :: device_ids(:)
        integer(c_intptr_t), target :: context, prog
        integer(c_intptr_t), target, intent(out) :: kernel
        character(len=1,kind=c_char), target :: source2(1:1000101), retinfo(1:1000000), c_options(1:1024), c_kernel_name(1:1024)
        type(c_ptr), target :: psource
        character(len=1024) :: options, kernel_name
        character(len=1,kind=c_char), allocatable, target :: source(:)
        integer :: i, irec, idevice, iplatform
        integer(c_size_t) :: iret


        psource=C_LOC(source) ! pointer to source code
        prog=clCreateProgramWithSource(context,1,C_LOC(psource),C_NULL_PTR,ierr)
        if (ierr.ne.CL_SUCCESS) stop 'clCreateProgramWithSource'
        !ierr= clGetProgramInfo(prog, CL_PROGRAM_SOURCE, sizeof(source2), C_LOC(source2), iret)
        !if (ierr.ne.CL_SUCCESS) stop 'clGetProgramInfo'
        !print '(a)','**** code retrieved from device start ****'
        !print '(1024a)',source2(1:min(iret,2048))
        !print '(a)','**** code retrieved from device end ****'


        !print '(///)'
         ! compiler options
        irec=len(trim(options))
        do i=1,irec
           c_options(i)=options(i:i)
        enddo
        c_options(irec+1)=C_NULL_CHAR
        ierr=clBuildProgram(prog,0, C_NULL_PTR,C_LOC(c_options),C_NULL_FUNPTR,C_NULL_PTR)
        !print *,ierr
        if (ierr.ne.CL_SUCCESS) then
           print *,'clBuildProgram',ierr
           ierr=clGetProgramBuildInfo(prog,device_ids(idevice), CL_PROGRAM_BUILD_LOG,sizeof(retinfo),C_LOC(retinfo),iret)
           if (ierr.ne.0) stop 'clGetProgramBuildInfo'
           print '(a)','build log start'
           print '(1024a)',retinfo(1:min(iret,1024))
           print '(a)','build log end'
           stop
        endif
        irec=len(trim(kernel_name))
        do i=1,irec
           c_kernel_name(i)=kernel_name(i:i)
        enddo
        c_kernel_name(irec+1)=C_NULL_CHAR
        kernel=clCreateKernel(prog,C_LOC(c_kernel_name),ierr)
        if (ierr.ne.0) stop 'clCreateKernel'


        ierr=clReleaseProgram(prog)
        if (ierr.ne.0) stop 'clReleaseProgram'

    end subroutine init_kernel

    subroutine read_file(filename, str, filesize)

        implicit none

        character(len=*),intent(in) :: filename
        character(len=1,kind=c_char),allocatable,intent(out) :: str(:)

        !local variables:
        integer :: iunit,istat,filesize
        character(len=1) :: c

        open(newunit=iunit,file=filename,status='OLD',&
                form='UNFORMATTED',access='STREAM',iostat=istat)

        if (istat==0) then

            !how many characters are in the file:
            inquire(file=filename, size=filesize)
            if (filesize>0) then

                !read the file all at once:
                allocate(str(filesize + 1) )
                read(iunit,pos=1,iostat=istat) str(:filesize)

                if (istat==0) then
                    !make sure it was all read by trying to read more:
                    read(iunit,pos=filesize+1,iostat=istat) c
                    if (.not. IS_IOSTAT_END(istat)) &
                        write(*,*) 'Error: file was not completely read.'
                else
                    write(*,*) 'Error reading file.'
                end if

                close(iunit, iostat=istat)
            else
                write(*,*) 'Error getting file size.'
            end if
        else
            write(*,*) 'Error opening file.'
        end if

    end subroutine read_file

    subroutine create_device_context(iplatform, platform_ids, idevice, device_ids, context, cmd_queue)
        ! Variable definitions for OpenCL API and in general
        integer(c_int32_t) :: err
        integer(c_size_t) :: zero_size = 0
        integer(c_size_t) :: temp_size
        integer(c_int) :: num_platforms
        integer(c_int) :: num_devices
        integer(c_intptr_t), allocatable, target, intent(out) :: platform_ids(:)
        integer(c_intptr_t), allocatable, target, intent(out) :: device_ids(:)

        integer(c_intptr_t), target :: ctx_props(3)
        integer(c_intptr_t), target, intent(out) :: context
        integer(c_int64_t) :: cmd_queue_props
        integer(c_intptr_t), target, intent(out) :: cmd_queue
        integer :: iplatform, idevice

        ! Get the number of platforms, prior to allocating an array.
        err = clGetPlatformIDs(0, C_NULL_PTR, num_platforms)
        if (err /= CL_SUCCESS) then
            print *, 'Error quering platforms: ', err, num_platforms
            call exit(1)
        end if

        if (num_platforms == 0) then
            print *, 'No platforms found'
            call exit(0)
        end if

        ! Allocate an array to hold platform handles.
        allocate(platform_ids(num_platforms))

        ! Get platforms IDs.
        err = clGetPlatformIDs(num_platforms, C_LOC(platform_ids), num_platforms)
        if (err /= CL_SUCCESS) then
            print *, 'Error getting platforms: ', err
            call exit(1)
        end if

        !
        ! Check number of devices for first platform.
        !
        err = clGetDeviceIDs(platform_ids(iplatform), CL_DEVICE_TYPE_ALL, 0, C_NULL_PTR, &
                num_devices)
        if (err /= CL_SUCCESS) then
            print *, 'Error quering devices: ', err
            call exit(1)
        end if

        if (num_devices == 0) then
            print *, 'No GPU devices found'
            call exit(0)
        end if

        ! Allocate an array to hold device handles.
        allocate(device_ids(num_devices))

        ! Get device IDs.
        err = clGetDeviceIDs(platform_ids(iplatform), CL_DEVICE_TYPE_ALL, num_devices, &
                C_LOC(device_ids), num_devices)
        if (err /= CL_SUCCESS) then
            print *, 'Error gettings devices: ', err
            call exit(1)
        end if

        !
        ! Create a context and a command queue.
        !

        ! Context.
        num_devices = 1
        ctx_props(1) = CL_CONTEXT_PLATFORM
        ctx_props(2) = platform_ids(iplatform)
        ctx_props(3) = 0

        context = clCreateContext(C_LOC(ctx_props), num_devices, &
                    C_LOC(device_ids), C_NULL_FUNPTR, C_NULL_PTR, err)

        if (err /= CL_SUCCESS) then
            print *, 'Error creating context: ', err
            call exit(1)
        end if

        ! Command queue.
        cmd_queue_props = 0
        cmd_queue = clCreateCommandQueue(context, device_ids(idevice), cmd_queue_props, err)

        if (err /= CL_SUCCESS) then
            print *, 'Error creating command queue: ', err
            call exit(1)
        end if

        print '(A)', 'Successfuly create OpenCL context and queue'
    end subroutine create_device_context

    subroutine query_platform_info(platform_id)
        ! Input variable.
        integer(c_intptr_t), intent(in)         :: platform_id

        ! Helper variables to work with OpenCL API.
        integer(c_int32_t) :: err
        integer(c_size_t) :: zero_size = 0
        integer(c_size_t) :: temp_size
        ! For quering devices.
        integer(c_int64_t) :: device_type
        integer(c_int32_t) :: num_devices
        integer(c_int) :: i
        integer(c_intptr_t), allocatable, target :: device_ids(:)

        ! String arrays for holding platform details.
        character, allocatable, target :: platform_profile(:)
        character, allocatable, target :: platform_version(:)
        character, allocatable, target :: platform_name(:)
        character, allocatable, target :: platform_vendor(:)
        character, allocatable, target :: platform_extensions(:)

        ! String array for holding device name.
        character, allocatable, target :: device_name(:)
        ! Maximum compute units for device.
        integer(c_int32_t), target :: device_cu

        ! Profile.
        err = clGetPlatformInfo(platform_id, CL_PLATFORM_PROFILE, zero_size, C_NULL_PTR, temp_size)
        allocate(platform_profile(temp_size))
        err = clGetPlatformInfo(platform_id, CL_PLATFORM_PROFILE, temp_size, C_LOC(platform_profile), temp_size)
        print *, 'Profile: ', platform_profile
        deallocate(platform_profile)

        ! Version.
        err = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, zero_size, C_NULL_PTR, temp_size)
        allocate(platform_version(temp_size))
        err = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, temp_size, C_LOC(platform_version), temp_size)
        print *, 'Version: ', platform_version
        deallocate(platform_version)

        ! Name.
        err = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, zero_size, C_NULL_PTR, temp_size)
        allocate(platform_name(temp_size))
        err = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, temp_size, C_LOC(platform_name), temp_size)
        print *, 'Name: ', platform_name
        deallocate(platform_name)

        ! Vendor.
        err = clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, zero_size, C_NULL_PTR, temp_size)
        allocate(platform_vendor(temp_size))
        err = clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, temp_size, C_LOC(platform_vendor), temp_size)
        print *, 'Vendor: ', platform_vendor
        deallocate(platform_vendor)

        ! Extensions.
        !err = clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS, zero_size, C_NULL_PTR, temp_size)
        !allocate(platform_extensions(temp_size))
        !err = clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS, temp_size, C_LOC(platform_extensions), temp_size)
        !print *, 'Extensions: ', platform_extensions
        !deallocate(platform_extensions)

        !
        ! Print device information for this platform.
        !
        ! Get device count.
        !device_type = CL_DEVICE_TYPE_ALL
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, C_NULL_PTR, num_devices)

        if (err /= CL_SUCCESS .or. num_devices < 1) then
            print *, 'No devices found: ', err
            return
        end if

        !print *
        !print '(A, I2)', 'Num Devices: ', num_devices

        ! Allocate an array to hold device handles.
        allocate(device_ids(num_devices))

        ! Get device IDs.
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, num_devices, C_LOC(device_ids), num_devices)
        if (err /= CL_SUCCESS) then
            print *, 'Error quering devices: ', err
            return
        end if

        ! Loop over devices and print information.
        do i = 1, num_devices
            ! Maximum compute units.
            temp_size = 4
            err = clGetDeviceInfo(device_ids(i), CL_DEVICE_MAX_COMPUTE_UNITS, temp_size, C_LOC(device_cu), temp_size)

            ! Name.
            err = clGetDeviceInfo(device_ids(i), CL_DEVICE_NAME, zero_size, C_NULL_PTR, temp_size)
            allocate(device_name(temp_size))
            err = clGetDeviceInfo(device_ids(i), CL_DEVICE_NAME, temp_size, C_LOC(device_name), temp_size)

            ! Print brief device details.
            write (*, '(A,I2,A,I3,A)', advance='no') ' Device (#', i, ', Compute Units: ', device_cu, ') - '
            print *, device_name

            deallocate(device_name)
        end do
    end subroutine query_platform_info
end module clroutines
