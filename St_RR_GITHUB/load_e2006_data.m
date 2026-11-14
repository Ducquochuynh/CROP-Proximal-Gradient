function [A,y,source_file] = load_e2006_data(data_dir,partition,max_rows)
    if nargin<3, max_rows=inf; end
    if ~isfolder(data_dir), mkdir(data_dir); end
    if strcmp(partition,'train')
        basename='E2006.train';
    elseif strcmp(partition,'test')
        basename='E2006.test';
    else
        error('Partition must be train or test.');
    end
    source_file=fullfile(data_dir,basename);
    archive=[source_file '.bz2'];
    url=['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/' ...
         'datasets/regression/' basename '.bz2'];
    if ~isfile(source_file)
        if ~isfile(archive)
            fprintf('Downloading %s from LIBSVM (train archive approx. 180 MB)...\n',basename);
            try
                websave(archive,url);
            catch ME
                error('E2006:DownloadFailed', ...
                    ['Dataset download failed: %s\nPlease download manually: %s\n', ...
                     'and place it in %s'],ME.message,url,data_dir);
            end
        end
        fprintf('Decompressing %s using macOS bzip2...\n',archive);
        temp=[source_file '.part'];
        if isfile(temp), delete(temp); end
        command=sprintf('bzip2 -dc %s > %s',quote_shell(archive),quote_shell(temp));
        [status,msg]=system(command);
        if status~=0 || ~isfile(temp)
            if isfile(temp), delete(temp); end
            error('E2006:DecompressionFailed', ...
                'macOS bzip2 decompression failed: %s. Archive: %s',msg,archive);
        end
        d=dir(temp);
        if isempty(d) || d.bytes==0
            delete(temp);
            error('E2006:EmptyData','Decompression created an empty file.');
        end
        [success,message]=movefile(temp,source_file,'f');
        if ~success, error('E2006:MoveFailed','%s',message); end
    end

    f=fopen(source_file,'r');
    if f<0, error('E2006:ReadFailed','Cannot read %s',source_file); end
    cleaner=onCleanup(@()fclose(f)); %#ok<NASGU>
    n=150360;  % official E2006-tfidf feature dimension
    cap=1000000;
    I=zeros(cap,1,'uint32');
    J=zeros(cap,1,'uint32');
    V=zeros(cap,1);
    y=zeros(20000,1);
    nnz_done=0; rows=0;
    fprintf('Parsing %s into a MATLAB sparse matrix (max_rows=%g)...\n', ...
        basename,max_rows);
    while rows<max_rows
        line=fgetl(f);
        if ~ischar(line), break; end
        if isempty(strtrim(line)), continue; end
        fields=sscanf(strrep(line,':',' '),'%f');
        if isempty(fields), continue; end
        if mod(numel(fields)-1,2)~=0
            error('E2006:BadFormat','Malformed LIBSVM line %d.',rows+1);
        end
        rows=rows+1;
        if rows>numel(y), y(end+10000,1)=0; end
        y(rows)=fields(1);
        cols=fields(2:2:end);
        vals=fields(3:2:end);
        if any(cols<1) || any(cols>n) || any(cols~=floor(cols))
            error('E2006:BadIndex','Bad feature index at line %d.',rows);
        end
        count=numel(cols);
        while nnz_done+count>cap
            grow=max(cap,1000000);
            I(end+grow,1)=uint32(0);
            J(end+grow,1)=uint32(0);
            V(end+grow,1)=0;
            cap=cap+grow;
        end
        idx=nnz_done+(1:count);
        I(idx)=uint32(rows);
        J(idx)=uint32(cols);
        V(idx)=vals;
        nnz_done=nnz_done+count;
        if mod(rows,1000)==0
            fprintf('  parsed %d rows, %d nonzero feature entries\n',rows,nnz_done);
        end
    end
    if rows==0, error('E2006:NoData','No observations parsed from %s.',source_file); end
    y=y(1:rows);
    % MATLAB sparse expects numeric index vectors; conversions are local.
    A=sparse(double(I(1:nnz_done)),double(J(1:nnz_done)),V(1:nnz_done),rows,n);
    fprintf('Loaded %s: %d x %d, nnz=%d.\n',basename,rows,n,nnz(A));
end
