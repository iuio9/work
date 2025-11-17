package com.qczy.common.rar;


import net.sf.sevenzipjbinding.IInArchive;
import net.sf.sevenzipjbinding.SevenZip;
import net.sf.sevenzipjbinding.impl.RandomAccessFileInStream;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

public class RAR5Util {
    public static void main(String[] args) throws IOException {
        String rarDir = "C:\\Users\\c\\Desktop\\文档\\超大图片2\\img4.rar";
        String outDir = "C:\\Users\\c\\Desktop\\文档\\123456\\";
        unRar(new File(rarDir),outDir);
    }
    public static List<String> unRar(File rarPath, String dstDirectoryPath) throws IOException {
        IInArchive archive;
        RandomAccessFile randomAccessFile;
        // 第一个参数是需要解压的压缩包路径，第二个参数参考JdkAPI文档的RandomAccessFile
        //r代表以只读的方式打开文本，也就意味着不能用write来操作文件
        randomAccessFile = new RandomAccessFile(rarPath, "r");
        archive = SevenZip.openInArchive(null, // null - autodetect
                new RandomAccessFileInStream(randomAccessFile));
        int[] in = new int[archive.getNumberOfItems()];
        for (int i = 0; i < in.length; i++) {
            in[i] = i;
        }
        archive.extract(in, false, new ExtractCallback(archive,dstDirectoryPath ));
        archive.close();
        randomAccessFile.close();
        ///data/microService/data/offline_dzhd_pdf/pdf/1637724142062/
        System.out.println("解压目标文件夹为："+dstDirectoryPath);
        List<String> allFileList = getAllFile(dstDirectoryPath, false);

        ArrayList<String> resultFileList = new ArrayList<>();
        String startString;
        String endString;
        String fianllyString;
        for (String s : allFileList) {
            if(s.startsWith("/") || s.startsWith("\\")){
                startString = s.substring(0,s.lastIndexOf("\\"));
                endString = s.substring(s.lastIndexOf("\\")+1);
                fianllyString = startString+"\\"+endString;
            }else {
                //windows系统去掉盘符
                s =s.substring(2);
                startString = s.substring(0,s.lastIndexOf("\\"));
                endString = s.substring(s.lastIndexOf("\\")+1);
                fianllyString = startString+"/"+endString;
            }
            System.out.println("rar文件电子回单解压前缀为："+startString+"rar文件电子回单解压后缀为："+endString);
            //解决liunx路径出现//导致文件路径错误
            fianllyString = fianllyString.replaceAll("//","/");
            resultFileList.add(fianllyString);
        }
        System.out.println("rar电子回单解压文件路径为"+resultFileList);
        return resultFileList;
    }
    /**
     * 获取路径下的所有文件/文件夹
     * @param directoryPath 需要遍历的文件夹路径
     * @param isAddDirectory 是否将子文件夹的路径也添加到list集合中
     * @return
     */
    public static List<String> getAllFile(String directoryPath,boolean isAddDirectory) {
        List<String> list = new ArrayList<String>();
        File baseFile = new File(directoryPath);
        if (baseFile.isFile() || !baseFile.exists()) {
            return list;
        }
        File[] files = baseFile.listFiles();
        for (File file : files) {
            if (file.isDirectory()) {
                if(isAddDirectory){
                    list.add(file.getAbsolutePath());
                }
                list.addAll(getAllFile(file.getAbsolutePath(),isAddDirectory));
            } else {
                list.add(file.getAbsolutePath());
            }
        }
        return list;
    }

}
