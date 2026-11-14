function q=quote_shell(path)
    % POSIX-shell safe quotation for paths with spaces and apostrophes.
    apostrophe=char(39);
    seq=[apostrophe '"' apostrophe '"' apostrophe];
    q=[apostrophe strrep(char(path),apostrophe,seq) apostrophe];
end
