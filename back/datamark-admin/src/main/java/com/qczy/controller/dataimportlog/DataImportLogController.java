package com.qczy.controller.dataimportlog;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.result.Result;
import com.qczy.model.response.DataResponse;
import com.qczy.model.response.FileDetailsResponse;
import com.qczy.service.DataImportLogService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/9 15:39
 * @Description:
 */
@RestController
@Api(tags = "数据导入记录")
public class DataImportLogController {


    @Autowired
    private DataImportLogService dataImportLogService;


    @GetMapping("/selectImportList")
    @ApiOperation("根据id获取当前数据集的导入记录）")
    public Result selectImportList(@RequestParam long sonId) {
        return Result.ok(dataImportLogService.selectImportList(sonId));
    }


    @GetMapping("/selectImportFileList")
    @ApiOperation("获取导入的文件信息")
    public Result selectImportFileList(
            @RequestParam Integer page,
            @RequestParam Integer limit,
            @RequestParam Integer id) {
        Page<FileDetailsResponse> pageParam = new Page<>(page, limit);
        IPage<FileDetailsResponse> fileList = dataImportLogService.selectImportFileList(pageParam, id);
        return Result.ok(fileList);
    }

}
