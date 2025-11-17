package com.qczy.service.impl;


import cn.hutool.core.io.FileUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.qczy.common.result.Result;
import com.qczy.mapper.DataSonMapper;
import com.qczy.mapper.FileMapper;
import com.qczy.model.entity.DataSonEntity;
import com.qczy.model.entity.FileEntity;
import com.qczy.model.request.MarkFileRequest;
import com.qczy.service.MarkFileService;
import com.qczy.utils.ImageUtils;
import net.coobird.thumbnailator.Thumbnails;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.ObjectUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;


@Service
public class MarkFileServiceImpl implements MarkFileService {

    @Value("${upload.formalPath}")
    private String formalPath;

    @Autowired
    private DataSonMapper dataSonMapper;

    @Autowired
    private FileMapper fileMapper;

    @Override
    public Integer addMarkFile(String sonId, Integer fileId) {

        DataSonEntity entity = dataSonMapper.selectOne(
                new LambdaQueryWrapper<DataSonEntity>()
                        .eq(DataSonEntity::getSonId, sonId)
        );
        if (ObjectUtils.isEmpty(entity)) {
            throw new RuntimeException("数据集不存在！");
        }

        // 组织上传路径
        String filePath = formalPath + entity.getFatherId() + "/" + "v" + entity.getVersion() + "/" + "mark" + "/";
        File fileMkdir = new File(filePath);
        // 判断目录是否存在，不存在，则创建
        if (!fileMkdir.exists()) {
            fileMkdir.mkdirs();
        }

        FileEntity fileEntity = fileMapper.selectById(fileId);
        if (ObjectUtils.isEmpty(fileEntity)) {
            throw new RuntimeException("文件不存在！");
        }
        // 文件原始路径
        String sourcePath = fileEntity.getFdPath();
        // 文件保存路径
        // 组织文件名
        String fileName = fileEntity.getFdName().substring(0, fileEntity.getFdName().lastIndexOf(".")) + "(mark)" +
                fileEntity.getFdName().substring(fileEntity.getFdName().lastIndexOf("."));
        String destPath = filePath + fileName;
        String shrinkPath = null;
        try {
            // 拷贝图片
            Files.copy(Paths.get(sourcePath), Paths.get(destPath),StandardCopyOption.REPLACE_EXISTING);


            // 生成缩略图
            if (ImageUtils.isImage(destPath)) {
                shrinkPath = saveShrinkFile(
                        formalPath + entity.getFatherId() + "/" + "v" + entity.getVersion()
                        , destPath);
            }


        } catch (Exception e) {
            throw new RuntimeException("文件拷贝失败！");
        }

        // 拷贝文件信息
        FileEntity newFileEntity = new FileEntity();
        BeanUtils.copyProperties(entity, newFileEntity);
        newFileEntity.setFdName(fileName);
        newFileEntity.setFdPath(destPath);
        newFileEntity.setHttpFilePath(shrinkPath);
        fileMapper.insert(newFileEntity);

        return newFileEntity.getId();


    }


    // 文件缩放
    private String saveShrinkFile(
            String dirPath,
            String newFilePath) {
        try {
            File file = new File(newFilePath);
            if (!file.exists()) { // 判断文件是否穿在
                return null;
            }
            String newPathStr = dirPath + "/" + "shrin" + "/";
            File dirPathStr = new File(newPathStr);
            if (!dirPathStr.exists()) { //判断当前目录是否存在
                dirPathStr.mkdirs();
            }

            // 开始新增缩放
            Thumbnails.of(file).
                    //scalingMode(ScalingMode.BICUBIC).
                            scale(0.1). // 图片缩放80%, 不能和size()一起使用
                    outputQuality(1.0). // 图片质量压缩80%
                    toFile(newPathStr + file.getName());


            return newPathStr + file.getName();

        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }

}
