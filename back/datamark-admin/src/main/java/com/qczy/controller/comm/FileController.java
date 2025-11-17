package com.qczy.controller.comm;

import cn.hutool.core.io.FileUtil;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.result.Result;
import com.qczy.model.entity.FileEntity;
import com.qczy.model.entity.MarkInfoEntity;
import com.qczy.model.request.DataSonQueryRequest;
import com.qczy.model.response.DataMarkResponse;
import com.qczy.service.DataMarkService;
import com.qczy.service.FileService;
import com.qczy.utils.FileFormatSizeUtils;
import com.qczy.utils.StringUtils;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.multipart.commons.CommonsMultipartFile;

import java.io.*;
import java.util.Objects;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/22 20:01
 * @Description:
 */
@RestController
@RequestMapping("/algorithm/file")
@Api(tags = "算法==文件上传接口")
public class FileController {


    @Autowired
    private FileService fileService;


    /**
     * 保存标注信息
     */
    @PostMapping("/uploadFile")
    @ApiOperation("上传文件")
    public Result addAlgorithmFile( MultipartFile file){
        File file1 = null;
        try {
            file1 = new File(file.getOriginalFilename());

            file.transferTo(file1);

        } catch (IOException e) {
            e.printStackTrace();
        }
        String s = fileService.addAlgorithmFile(file1);
        if(s == null){
            return Result.fail();
        }else{
            return Result.ok(s);
        }

    }


}
