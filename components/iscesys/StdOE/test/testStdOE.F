       subroutine testStdOE
         character*256 filename,tag
         character*1 stdTypeRet 
          
         call writeStd("hello")
         filename = "testStdOut.log"
         call setStdOutFile(filename)
         tag = 'TestTag'
         call setStdOutFileTag(tag)
         filename = "testStdErr.log"
         call setStdErrFile(filename)
         call writeStdOut("first message")
         call writeStdErr("first message")
         call writeStdOut("second message")
         call writeStdErr("second message")
         call setStdOut("screen")
         call setStdErr("screen")
         call writeStdOut("first message")
         call writeStdErr("first message")
         call writeStdOut("second message")
         call writeStdErr("second message")
         filename = "test.log"
         call writeStdFile(filename,"first message")
         call writeStdFile(filename,"second message")
         call getStdOut(stdTypeRet)
         call writeStd(stdTypeRet)
         call getStdErr(stdTypeRet)
         call writeStd(stdTypeRet)
         call setStdOut("file")
         call setStdErr("file")
         call writeStdOut("third message")
         call writeStdErr("third message")
       end
