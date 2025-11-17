package com.qczy.controller.dataupload;

import com.qczy.common.constant.SystemConstant;
import com.qczy.common.result.Result;
import com.qczy.mapper.TempFileMapper;
import com.qczy.model.entity.TempFileEntity;
import com.qczy.model.request.DeleteRequest;
import com.qczy.model.response.TempFileResponse;
import com.qczy.utils.FileFormatSizeUtils;
import com.qczy.utils.ImageUtils;
import com.qczy.utils.URLUtils;
import io.swagger.annotations.Api;
import org.apache.ibatis.annotations.Param;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.ObjectUtils;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.awt.*;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Objects;


/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/7 15:11
 * @Description:
 */
@RestController
@Api(tags = "临时文件存储")
public class TempUploadController {

    private static final Logger log = LoggerFactory.getLogger(TempUploadController.class);


    @Value("${upload.port}")
    private String port;

    @Value("${upload.tempPath}")
    private String tempPath;

    @Autowired
    private TempFileMapper tempFileMapper;

    /**
     * 多文件上传
     */

    @Transactional
    @PostMapping("/temp/anyUpload")
    public Result anyUpload(@RequestPart(name = "file", required = false) MultipartFile file) throws IOException {


        if (!ImageUtils.isFile(tempPath + file.getOriginalFilename())) {
            return Result.fail("文件类型只能为jpg/png/bmp/jpeg/webp/json/xml");
        }

        if (file.getSize() >= 31457280L) {
            return Result.fail("文件大小不能超过30MB");
        }




        File fileMkdir = new File(tempPath);
        // 判断目录是否存在，不存在，则创建
        if (!fileMkdir.exists()) {
            fileMkdir.mkdirs();
        }


        InputStream in = null;
        FileOutputStream out = null;
        List<TempFileResponse> tempFileResponseList = new ArrayList<>();

        // 执行文件上传
        try {

            TempFileResponse tempFileResponse = new TempFileResponse();
            // 文件输入流
            in = file.getInputStream();
            // 文件输出流
            out = new FileOutputStream(tempPath + file.getOriginalFilename());


            // 一次读8192个字节
            byte[] buf = new byte[8192];
            int readLen = 0;

            while ((readLen = in.read(buf)) != -1) {
                // 读到数据以后，就写入
                out.write(buf, 0, readLen);
            }
            TempFileEntity tempFile = new TempFileEntity();
            tempFile.setFdTempPath(tempPath + file.getOriginalFilename());
            tempFile.setFdName(file.getOriginalFilename());
            tempFile.setFdType(file.getContentType());
            tempFile.setFdSuffix(Objects.requireNonNull(file.getOriginalFilename()).substring(file.getOriginalFilename().lastIndexOf(".")));
            tempFile.setFdAccessPath(file.getOriginalFilename());
            tempFile.setFdSize(FileFormatSizeUtils.formatSize(file.getSize()));
            tempFile.setCreateTime(new Date());
            // 判断是否上传的是图片，如果为图片，则获取宽高
            if (ImageUtils.isImage(tempFile.getFdTempPath())) {
                if (ImageUtils.isImageCorrupted(tempFile.getFdTempPath())) {
                    return Result.fail("无法解析图片，可能已损坏！");
                }
                int[] ints = ImageUtils.getImageDimensions(tempFile.getFdTempPath());
                tempFile.setWidth(ints[0]);
                tempFile.setWidth(ints[1]);
            }
            int result = tempFileMapper.insertTempFile(tempFile);
            if (result < SystemConstant.MAX_SIZE) {
                throw new RuntimeException("后端异常，临时文件数据库新增失败！");
            }

            tempFileResponse.setId(tempFile.getId());
            tempFileResponse.setName(tempFile.getFdName());
            tempFileResponse.setPath(URLUtils.encodeURL(port + file.getOriginalFilename()));
            tempFileResponseList.add(tempFileResponse);


        } catch (RuntimeException e) {
            log.error("-----------------Upload File Fail-----------------");
            throw new RuntimeException(e);

        } finally {
            if (in != null) {
                in.close();
            }
            if (out != null) {
                out.close();
            }
        }

        log.info("-----------------Upload File Successfully-----------------");
        return Result.ok(tempFileResponseList);


    }


    @PostMapping("/temp/deleteTempFile")
    public Result deleteTempFile(@RequestBody DeleteRequest request) throws IOException {
        if (ObjectUtils.isEmpty(request)) {
            return Result.fail("上传的数据为空！");
        }

        for (Integer filId : request.getIds()) {
            TempFileEntity tempFile = tempFileMapper.selectById(filId);
            if (ObjectUtils.isEmpty(tempFile)) {
                return Result.fail("临时文件对象获取失败");
            }
            File file = new File(tempFile.getFdTempPath());
            if (!file.exists()) {
                return Result.fail("文件删除失败");
            }
            file.delete();
        }
        return Result.ok();
    }


}
