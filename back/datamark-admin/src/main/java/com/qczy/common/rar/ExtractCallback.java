package com.qczy.common.rar;

import net.sf.sevenzipjbinding.*;

import java.io.*;

public class ExtractCallback implements IArchiveExtractCallback {

        private int index;
        private IInArchive inArchive;
        private String ourDir;

        public ExtractCallback(IInArchive inArchive, String ourDir) {
            this.inArchive = inArchive;
            this.ourDir = ourDir;
        }

        @Override
        public void setCompleted(long arg0) throws SevenZipException {
        }

        @Override
        public void setTotal(long arg0) throws SevenZipException {
        }

        @Override
        public ISequentialOutStream getStream(int index, ExtractAskMode extractAskMode) throws SevenZipException {
            this.index = index;
            final String path = (String) inArchive.getProperty(index, PropID.PATH);
            final boolean isFolder = (boolean) inArchive.getProperty(index, PropID.IS_FOLDER);
            final String[] oldPath = {""};
            return new ISequentialOutStream() {
                public int write(byte[] data) throws SevenZipException {
                    try {
                        if (!isFolder) {
//                        System.out.println(path);
                            File file = new File(ourDir+"\\" + path);
                            if (path.equals(oldPath[0])){
                                save2File(file, data,true);
                            }else{
                                save2File(file, data,false);
                            }
                            oldPath[0] = path;
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    return data.length;
                }
            };
            /*return data -> {
                try {
                    if (!isFolder) {
                        File file = new File(ourDir + path);
                        save2File(file, data);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
                return data.length;
            };*/

        }

        @Override
        public void prepareOperation(ExtractAskMode arg0) throws SevenZipException {
        }

        @Override
        public void setOperationResult(ExtractOperationResult extractOperationResult) throws SevenZipException {

        }
        //解决字节丢失  未验证
        public boolean save2File(File file, byte[] msg,boolean append) {
            OutputStream fos = null;
            try {
                File parent = file.getParentFile();
                boolean bool;
                if ((!parent.exists()) && (!parent.mkdirs())) {
                    return false;
                }
    //            fos = new FileOutputStream(file);
                fos = new FileOutputStream(file,append);//是否追加
                fos.write(msg);
                fos.flush();
                return true;
            } catch (FileNotFoundException e) {
                return false;
            } catch (IOException e) {
                File parent;
                return false;
            } finally {
                if (fos != null) {
                    try {
                        fos.close();
                    } catch (IOException e) {
                    }
                }
            }
        }

        public static boolean save2File(File file, byte[] msg) {
            OutputStream fos = null;
            try {
                File parent = file.getParentFile();
                if ((!parent.exists()) && (!parent.mkdirs())) {
                    return false;
                }
                fos = new FileOutputStream(file, true);
                fos.write(msg);
                fos.flush();
                return true;
            } catch (FileNotFoundException e) {
                e.printStackTrace();
                return false;
            } catch (IOException e) {
                e.printStackTrace();
                return false;
            } finally {
                try {
                    fos.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }


